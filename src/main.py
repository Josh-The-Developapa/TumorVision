from torch import nn, optim
import torch
from utils import (
    LoadImageData,
    visualise_batch_images,
    train_model,
    test_model,
    load_model,
)
from model import TVRN50

# Set the device to 'cuda' if available (for Nvidia GPUs running CUDA), else 'cpu'
device = "cuda" if torch.cuda.is_available() else "cpu"

# Set the device to 'mps' if available (for Macbooks with Apple Silicon), else 'cpu'
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load the training and testing data using LoadImageData function
train_dataloader, test_dataloader = LoadImageData(
    "../data",  # Path to the dataset
    batch_size=4,  # Batch size for data loading
)

# Visualise a batch of images from the Brain Tumor dataset
visualise_batch_images(2, train_dataloader, 4)

# Define a new instance of the TVRN50 model and move it to the appropriate device
model = TVRN50().to(device=device)

# Optionally, load a pre-trained model from a file
model = load_model(model_class=TVRN50, file="../models/TVRN50.pt").to(device=device)

# Calculate and print the total number of parameters in the model
total_params = sum(p.numel() for p in model.parameters())
print("TumorVision ResNet50 (TVRN50)")
print(f"Total parameters: {total_params:,} parameters")

# Define the loss function (criterion) as Binary Cross Entropy with Logits
criterion = nn.BCEWithLogitsLoss().to(device=device)

# Define the optimizer as Stochastic Gradient Descent with a learning rate of 0.0001
optimiser = optim.SGD(model.parameters(), lr=0.0001)

# Define a learning rate scheduler that reduces the learning rate when a metric has stopped improving
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimiser, mode="min", factor=0.1, patience=2
)

# Test the model before training to get the initial performance
# print("Testing before training")
test_model(model, criterion, test_dataloader, device=device)

# Train the model
train_model(
    model,
    criterion,
    optimiser,
    scheduler=scheduler,
    dataloader=train_dataloader,
    val_dataloader=test_dataloader,
    epochs=10,  # Number of epochs to train for
    device=device,
)

# Test the model after training to evaluate its performance
print("Testing after training")
test_model(model, criterion, test_dataloader, device=device)

# TVRN50
# Actual training results to capacity:
# Accuracy: 47/51  | 92.16%  | Average Loss: 0.474
