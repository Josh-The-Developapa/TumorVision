import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class FERDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Define your classes (assuming FER-13 classes)
        self.classes = (
            "Angry",
            "Disgust",
            "Fear",
            "Happy",
            "Sad",
            "Surprise",
            "Neutral",
        )

        self.data = self.load_dataset()

    def load_dataset(self):
        data = []
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                data.append((img_path, class_idx))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label_index = self.data[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # One-hot encode the label
        label = torch.zeros(len(self.classes), dtype=torch.float)
        label[label_index] = 1.0

        return image, label
