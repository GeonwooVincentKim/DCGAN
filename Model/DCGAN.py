from __future__ import print_function
import argparse
import os
import random

import torch
from torch import nn, optim, backends
from torch.utils.data import TensorDataset, DataLoader

from torchvision import datasets, transforms
from torchvision.utils import save_image

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import import_ipynb

EPOCHS = 500
BATCH_SIZE = 100
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print("Current Device : ", DEVICE)

# Set random seed for reproductibility
manualSeed = 999
print("Random Seed : ", manualSeed)
random.seed(manualSeed)


dataroot = '../data/'
image_size = 64
batch_size = 128
workers = 2

train_set = datasets.ImageFolder(
    root=dataroot,
    transform=transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
)

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True, num_workers=workers
)

real_batch = next(iter(train_loader))
plt.figure(figsize=(10, 10))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(DEVICE))))
