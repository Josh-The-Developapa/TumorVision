import torch
from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F


class BrainTumourDataset(Dataset):
    """
    Custom dataset for 224x224 MRI scans of human brains for multi-class classification.
    This dataset is designed for use with a PyTorch DataLoader.

    Labels:
    0 - Healthy
    1 - Tumor
    """

    def __init__(self, dataset, transform):
        """
        Initialize the dataset with image-label pairs and transformation operations.

        Args:
            dataset (list of tuples): List containing tuples of image file paths and their corresponding labels.
            transform (callable): Transform to be applied on an image sample.
        """
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Total number of samples.
        """
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Retrieve an image and its corresponding label at a specified index.

        Args:
            index (int): Index of the sample to be retrieved.

        Returns:
            tuple: (transformed image tensor, label tensor)
        """
        # Convert the file path to a PIL image and ensure it's in RGB format.
        imagePIL = Image.open(self.dataset[index][0]).convert("RGB")

        # Apply the specified transformations to the image (e.g., resize, normalize, etc.).
        image = self.transform(imagePIL)

        # Convert the label to a tensor of type float32.
        label = torch.tensor(self.dataset[index][1]).type(torch.float32)

        return image, label
