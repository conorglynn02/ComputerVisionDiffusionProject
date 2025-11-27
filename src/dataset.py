import os
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.utils import save_image

from PIL import Image


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

##########################################################
# 1. SAVE ORIGINAL STL-10 IMAGES
##########################################################
def save_stl10_original(root="data/stl10_original"):
    ensure_dir(root)

    transform = transforms.ToTensor()

    # STL10 has train & test splits
    stl_train = datasets.STL10(root="data", split="train", download=True, transform=transform)
    stl_test  = datasets.STL10(root="data", split="test",  download=True, transform=transform)

    all_data = torch.utils.data.ConcatDataset([stl_train, stl_test])

    for idx, (img, label) in enumerate(tqdm(all_data, desc="Saving original STL-10")):
        class_dir = os.path.join(root, str(label))
        ensure_dir(class_dir)
        save_path = os.path.join(class_dir, f"{idx:05d}.png")
        save_image(img, save_path)

##########################################################
# 2. STYLE TRANSFORM HELPERS
##########################################################
def to_cv2(img_tensor):
    img = img_tensor.numpy().transpose(1, 2, 0)
    img = (img * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img_bgr

def from_cv2(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1)
    return img_tensor

##########################################################
# 3. STYLE FUNCTIONS (Sketch, Cartoon, Watercolor)
##########################################################
def sketch_style(img_tensor):
    img = to_cv2(img_tensor)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray_blur, 50, 150)
    edges_inv = cv2.bitwise_not(edges)
    sketch = cv2.cvtColor(edges_inv, cv2.COLOR_GRAY2BGR)
    return from_cv2(sketch)

def cartoon_style(img_tensor):
    img = to_cv2(img_tensor)
    color = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2
    )
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cartoon = cv2.bitwise_and(color, edges)
    return from_cv2(cartoon)

def watercolor_style(img_tensor):
    img = to_cv2(img_tensor)
    blur = cv2.GaussianBlur(img, (7, 7), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[...,1] *= 1.2
    hsv[...,2] *= 1.1
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    water = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return from_cv2(water)

STYLE_FUNCS = {
    "sketch": sketch_style,
    "cartoon": cartoon_style,
    "watercolor": watercolor_style,
}

##########################################################
# 4. GENERATE STYLISED STL-10
##########################################################
def generate_stylised_stl10(
    original_root="data/stl10_original",
    out_root="data",
    styles=("sketch", "cartoon", "watercolor"),
):
    transform = transforms.ToTensor()

    stl_train = datasets.STL10(root="data", split="train", download=False, transform=transform)
    stl_test  = datasets.STL10(root="data", split="test",  download=False, transform=transform)
    all_data = torch.utils.data.ConcatDataset([stl_train, stl_test])

    for style in styles:
        style_func = STYLE_FUNCS[style]
        style_root = os.path.join(out_root, f"stl10_{style}")
        print(f"Creating style: {style} -> {style_root}")

        for idx, (img, label) in enumerate(tqdm(all_data, desc=f"Stylising {style}")):
            class_dir = os.path.join(style_root, str(label))
            ensure_dir(class_dir)
            styled_img = style_func(img)
            save_path = os.path.join(class_dir, f"{idx:05d}.png")
            save_image(styled_img, save_path)



STYLE_TO_ID = {
    "original": 0,
    "sketch": 1,
    "cartoon": 2,
    "watercolor": 3,
}

class MultiStyleSTL10(Dataset):
    def __init__(self, data_root="data",
                 styles=("original", "sketch", "cartoon", "watercolor"),
                 transform=None):
        self.samples = []
        self.transform = transform

        for style in styles:
            if style == "original":
                style_dir = "stl10_original"
            else:
                style_dir = f"stl10_{style}"

            style_path = os.path.join(data_root, style_dir)
            style_id = STYLE_TO_ID[style]

            if not os.path.isdir(style_path):
                print(f"[WARN] Missing folder: {style_path}, skipping")
                continue

            # class folders: 0..9
            for class_name in os.listdir(style_path):
                class_dir = os.path.join(style_path, class_name)
                if not os.path.isdir(class_dir):
                    continue
                label = int(class_name)

                for fname in os.listdir(class_dir):
                    if not fname.endswith(".png"):
                        continue
                    self.samples.append({
                        "path": os.path.join(class_dir, fname),
                        "label": label,
                        "style": style_id,
                    })

        print(f"[INFO] Loaded {len(self.samples)} images from styles={styles}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        meta = self.samples[idx]
        img = Image.open(meta["path"]).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        label = meta["label"]
        style = meta["style"]

        return img, label, style