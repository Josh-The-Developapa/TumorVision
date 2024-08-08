import torch
import torchvision
from torchvision import models
from torch import nn
import torch.nn.functional as F


class TVRN50(nn.Module):
    """Class for TumourVision ResNet50 (TVRN50), a powerful Convolutional Neural Network (CNN) designed
    to classify brain MRI scans for brain tumor detection.
    The model is built using PyTorch and leverages Torch, TorchVision, and Matplotlib
    libraries to achieve accurate and insightful results.

    TVRN50 has 23.64 million parameters. 23,643,393 parameters to be exact.
    """

    def __init__(
        self, num_classes=1
    ):  # Set num_classes to 1 (can be updated as needed)
        super(TVRN50, self).__init__()
        # Load the pre-trained ResNet50 model from torchvision
        self.resnet50 = models.resnet50(
            weights=torchvision.models.ResNet50_Weights.DEFAULT
        )

        # Ensure all parameters of the ResNet50 model are set to be trainable
        for param in self.resnet50.parameters():
            param.requires_grad = True

        # Get the number of input features of the original fully connected (fc) layer of ResNet50
        n_inputs = self.resnet50.fc.in_features

        # Redefine the fully connected layer (fc) to match the classification problem
        self.resnet50.fc = nn.Sequential(
            nn.Linear(n_inputs, 64),  # First linear layer (fully connected layer)
            nn.ReLU(),  # Activation function (Rectified Linear Unit)
            nn.Dropout(p=0.4),  # Dropout layer to prevent overfitting
            nn.Linear(64, 64),  # Second linear layer
            nn.ReLU(),  # Activation function
            nn.Dropout(p=0.4),  # Dropout layer
            nn.Linear(64, num_classes),  # Final linear layer for output
        )

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after passing through the ResNet50 model.
        """
        return self.resnet50(x)


# Example instantiation of the model
# model = TVRN50(num_classes=4)  # Set num_classes to the desired number of output classes
