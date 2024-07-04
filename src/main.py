import torch
import torch.nn as nn
import torchvision
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
import pandas
from utils import (
    LoadImageData,
    visualise_batch_images,
    matplotlib_imshow,
    train_model,
    test_model,
    save_model,
    load_model,
)
from model import EmotiNet
from PIL import Image


train_dataloader, test_dataloader = LoadImageData("../data")

visualise_batch_images(dataloader=test_dataloader, batch_num=20, num_images=4)

model = EmotiNet()
criterion = nn.CrossEntropyLoss()


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_model(model, criterion, optimizer, train_dataloader, test_dataloader, epochs=2)

test_model(model, criterion, test_dataloader)

# save_model(model, "../models/VGG16.pt")
