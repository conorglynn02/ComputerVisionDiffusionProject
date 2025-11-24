import os
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.utils import save_image


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def save_cifar10_original(root="data/cifar10_original"):
    ensure_dir(root)

    transform = transforms.ToTensor()
    cifar10_train = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
    cifar10_test = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)

    all_data = torch.utils.data.ConcatDataset([cifar10_train, cifar10_test])

    for idx, (img, label) in enumerate(tqdm(all_data, desc="Saving original CIFAR-10 images")):
        class_dir = os.path.join(root, str(label))
        ensure_dir(class_dir)
        save_path = os.path.join(class_dir, f"{idx:05d}.png")
        save_image(img, save_path)
