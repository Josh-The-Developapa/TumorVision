import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split
from dataset import BrainTumourDataset
import os
import matplotlib.pyplot as plt
import datetime


def blurt(accuracy: float, loss: float):
    """Function to have our system speak out the accuracy and average loss of the model"""
    os.system(
        f'say "Testing complete. Your Model has a {100*accuracy:.2f}% Accuracy, and an Average Loss of {loss:.3f}"'
    )


def calculate_mean_std(dataset, batch_size: int, device: torch.device):
    # Create a data loader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    mean = torch.zeros(3, device=device)
    std = torch.zeros(3, device=device)
    total_images = 0

    for images, _ in data_loader:
        images = images.to(device)
        images = images.view(images.size(0), images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += images.size(0)

    mean /= total_images
    std /= total_images

    return mean, std


def LoadImageData(root: str, batch_size: int):
    """Function to process and load our data \n\n
    Returns the test and train dataloaders\n\n
    Each 'class' must have a subfolder inside the root, "data" folder. So data/NoTumour & data/HasTumour
    """
    # ImageFolder dataset containing paths and labels of images.
    dataset = datasets.ImageFolder(root)

    mean, std = calculate_mean_std(
        (
            BrainTumourDataset(
                dataset.imgs,
                transform=transforms.Compose(
                    [
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                    ]
                ),
            )
        ),
        batch_size=4,
        device="mps",
    )

    # The transforms for our dataset
    transform = transforms.Compose(
        [
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            # transforms.RandomRotation(10),
            # transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.Resize((224, 224)),
            # transforms.ColorJitter(
            #     brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
            # ),
            # transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    # Split our data into train and test data and labels
    train_data, test_data, train_labels, test_labels = train_test_split(
        dataset.imgs, dataset.targets, test_size=0.2, random_state=42
    )

    train_dataset = BrainTumourDataset(train_data, transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = BrainTumourDataset(test_data, transform)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader


def matplotlib_imshow(batch: list, num_images: int):
    """Function for producing an inline display
    of a set number of images in a given batch"""

    classes = ("Healthy", "Tumor")

    # Fetch only the first (num_images)th
    batch = [(batch[0][0:num_images]), batch[1][0:num_images]]

    fig, axes = plt.subplots(1, len(batch[0]), figsize=(10, 5))
    for idx, img in enumerate(batch[0]):
        # Assuming img is normalized with mean and std
        # Adjust this based on your normalization specifics
        imgu = img.permute(1, 2, 0).cpu().numpy()

        # Unnormalize if needed (modify these values based on your normalization)
        # Example: if your images are normalized with mean and std
        # imgu = imgu * std + mean
        # This example assumes mean=0 and std=1 for simplicity

        # Clip image values to [0, 1] if they're normalized floats
        imgu = (imgu - imgu.min()) / (imgu.max() - imgu.min())

        ax = axes[idx]
        ax.set_title(classes[batch[1][idx].type(torch.int64)])
        ax.imshow(imgu)

    plt.tight_layout()
    plt.show()


def visualise_batch_images(batch_num: int, dataloader: DataLoader, num_images: int):
    """Function that calls `matplotlib_imshow()` iteratively to produce an inline display for the batch specified\n\n
    Eliminates the need of calling matplotlib_imshow in a for loop
    """
    i = 0
    for idx, batch in enumerate(dataloader):
        if idx == (batch_num - 1):
            matplotlib_imshow(batch, num_images)
            break


def train_model(
    model: torch.nn.Module,
    criterion,
    optimizer,
    dataloader: DataLoader,
    val_dataloader: DataLoader,
    epochs: int,
    device: torch.device,
    scheduler,
):
    """A function to train our model and evaluate it after each epoch.

    It passes the entire dataset from a dataloader through the model for a specified number of epochs,
    and evaluates the model on a validation dataset after each epoch.
    """
    model.train()
    start = datetime.datetime.now()

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        print(f"\n\n Epoch: {epoch}\n\n -----------------------")
        running_train_loss = 0.0
        num_train_batches = len(dataloader)

        for idx, batch in enumerate(dataloader):
            imgs, labels = batch[0].to(device), batch[1].to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            predictions = model(imgs).squeeze()

            # Calculate loss
            loss = criterion(predictions, labels.squeeze())

            # Back propagation and update parameters
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

            if idx % 10 == 0:
                print(f"Loss: {loss:.3f} | Batch: {idx}/{len(dataloader)}")

        average_train_loss = running_train_loss / num_train_batches
        train_losses.append(average_train_loss)
        print(f"Average Training Loss for Epoch {epoch}: {average_train_loss:.3f}")

        # Evaluation loop
        model.eval()
        running_val_loss = 0.0
        num_val_batches = len(val_dataloader)

        with torch.no_grad():
            for val_batch in val_dataloader:
                val_imgs, val_labels = val_batch[0].to(device), val_batch[1].to(device)

                # Forward pass
                val_predictions = model(val_imgs).squeeze()

                # Calculate loss
                val_loss = criterion(val_predictions, val_labels)
                running_val_loss += val_loss.item()

        average_val_loss = running_val_loss / num_val_batches
        val_losses.append(average_val_loss)
        print(f"Average Validation Loss for Epoch {epoch}: {average_val_loss:.3f}")

        # Step the scheduler with the validation loss
        scheduler.step(average_val_loss)
        print(f"Learning Rate after Epoch {epoch}: {scheduler.get_last_lr()[0]}")

        # Save the model after each epoch
        save_model(model, f"../models/TV-Epoch-{epoch+1}.pt")

        # Switch back to training mode
        model.train()

    end = datetime.datetime.now()

    # Time taken for the specified number of epochs
    run_time = end - start
    print(f"Run time: {run_time}")

    os.system(f'say "Training complete!"')

    # Plotting the loss graph
    plt.figure()
    plt.plot(range(epochs), train_losses, label="Training Loss")
    plt.plot(range(epochs), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss vs. Epochs")
    plt.legend()
    plt.show()


def test_model(
    model: torch.nn.Module, criterion, dataloader: DataLoader, device: torch.device
):
    """Function to evaluate our model's performance after training.

    It iterates over data it has never seen before.
    """
    start = datetime.datetime.now()
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for images, labels in dataloader:

            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss = criterion(output.squeeze(), labels.float())

            test_loss += loss.item()
            predicted = torch.round(
                torch.sigmoid(output.squeeze())
            )  # Convert output to probabilities and then to binary
            # Traverse through each prediction and compare with label
            for i in range(len(predicted)):

                if predicted[i] == labels[i]:
                    correct += 1

    accuracy = correct / len(dataloader.dataset)
    avg_loss = test_loss / len(dataloader)

    end = datetime.datetime.now()

    # Time taken for the specified number of epochs
    run_time = end - start
    print(f"Run time: {run_time}")

    print(
        f"Accuracy: {correct}/{len(dataloader.dataset)}  | {100*accuracy:.2f}%  | Average Loss: {avg_loss:.3f}"
    )
    blurt(accuracy, loss=avg_loss)


def save_model(model: torch.nn.Module, file: str):
    """Function to save a given model's parameters in the specified file path"""
    torch.save(model.state_dict(), file)


def load_model(model_class, file: str):
    """Function to load a given model's parameters in the specified file path"""
    loaded_model = model_class()
    loaded_model.load_state_dict(torch.load(file))
    return loaded_model
