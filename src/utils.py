import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torchvision
from dataset import FERDataset
import datetime
import os


def blurt(accuracy: float, loss: float):
    """Function to have our system speak out the accuracy and average loss of the model"""
    os.system(
        f'say "Testing complete. Your Model has a {100*accuracy:.2f}% Accuracy, and an Average Loss of {loss:.3f}"'
    )


# Define a function to load image data
def LoadImageData(root: str):
    """
    Load image data from the specified root directory and return training and test DataLoaders.

    Parameters:
        root (str): The root directory where the training and test datasets are located.

    Returns:
        train_dataloader (DataLoader): The DataLoader for the training dataset with a batch size of 64 and shuffling enabled.
        test_dataloader (DataLoader): The DataLoader for the test dataset with a batch size of 64 and shuffling enabled.
    """
    # Define transformations
    transform = transforms.Compose(
        [
            transforms.Resize((48, 48)),
            transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
            transforms.ToTensor(),  # Convert images to PyTorch tensors
        ]
    )

    # Initialize custom datasets
    train_dataset = FERDataset(root + "/train", transform=transform)
    test_dataset = FERDataset(root + "/test", transform=transform)

    # Create DataLoaders
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    return train_dataloader, test_dataloader


def matplotlib_imshow(batch, num_images):
    """Function for producing an inline display of a set number of images in a given batch"""
    classes = (
        "Angry",
        "Disgust",
        "Fear",
        "Happy",
        "Sad",
        "Surprise",
        "Neutral",
    )

    images, labels = batch
    images = images[:num_images]
    labels = labels[:num_images]

    fig, axes = plt.subplots(1, len(images), figsize=(10, 5))
    for idx, img in enumerate(images):
        ax = axes[idx]
        actual_label = torch.argmax(labels[idx]).item()
        ax.set_title(classes[actual_label])
        ax.imshow(img.permute(1, 2, 0))

        print(f"Image {idx + 1} - Actual Label: {classes[actual_label]}")

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
    optimiser,
    dataloader: DataLoader,
    val_dataloader: DataLoader,
    epochs: int,
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
            imgs, labels = batch[0], batch[1]

            # Zero gradients
            optimiser.zero_grad()

            # Forward pass
            predictions = model(imgs).squeeze()

            # Calculate loss
            loss = criterion(predictions, labels.squeeze())

            # Back propagation and update parameters
            loss.backward()
            optimiser.step()

            running_train_loss += loss.item()

            if idx % 100 == 0:
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
                val_imgs, val_labels = val_batch[0], val_batch[1]

                # Forward pass
                val_predictions = model(val_imgs).squeeze()

                # Calculate loss
                val_loss = criterion(val_predictions, val_labels)
                running_val_loss += val_loss.item()

        average_val_loss = running_val_loss / num_val_batches
        val_losses.append(average_val_loss)
        print(f"Average Validation Loss for Epoch {epoch}: {average_val_loss:.3f}")

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


def test_model(model: torch.nn.Module, criterion, dataloader: DataLoader):
    """Function to evaluate our model's performance after training \n\n
    Having it iterate over data it has never seen before"""

    start = datetime.datetime.now()
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for images, labels in dataloader:
            output = model(images)
            test_loss += criterion(output.squeeze(), labels)

            if torch.argmax(output) == torch.argmax(labels):
                correct += 1

    accuracy = correct / len(dataloader)
    avg_loss = test_loss / len(dataloader)

    end = datetime.datetime.now()

    # Time taken for the specified number of epochs
    run_time = end - start
    print(f"Run time: {run_time}")

    print(
        f"Accuracy: {correct}/{ len(dataloader)}  | {100*accuracy:.2f} %  | Average Loss: {avg_loss:.3f}"
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
