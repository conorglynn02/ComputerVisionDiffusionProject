import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from diffusers import DDPMScheduler

from dataset import MultiStyleSTL10
from model import ClassStyleConditionedUNet


def get_dataloaders(
    data_root="data",
    batch_size=16,
):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = MultiStyleSTL10(
        data_root=data_root,
        styles=("original", "sketch", "cartoon", "watercolor"),
        transform=transform,
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return loader


def train(
    data_root="data",
    image_size=96,
    num_epochs=5,
    batch_size=16,
    lr=1e-4,
    device="cuda" if torch.cuda.is_available() else "cpu",
    save_dir="checkpoints",
):
    os.makedirs(save_dir, exist_ok=True)

    dataloader = get_dataloaders(data_root=data_root, batch_size=batch_size)

    model = ClassStyleConditionedUNet(
        num_classes=10,
        num_styles=4,
        image_size=image_size,
    ).to(device)

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in pbar:
            imgs, class_labels, style_labels = batch
            imgs = imgs.to(device)
            class_labels = class_labels.to(device)
            style_labels = style_labels.to(device)

            # normalize images to [-1, 1]
            imgs = 2 * imgs - 1.0

            # sample random timesteps
            bsz = imgs.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (bsz,), device=device
            ).long()

            # add noise
            noise = torch.randn_like(imgs)
            noisy_imgs = noise_scheduler.add_noise(imgs, noise, timesteps)

            # predict the noise with our conditional UNet
            noise_pred = model(noisy_imgs, timesteps, class_labels, style_labels)

            loss = mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            pbar.set_postfix({"loss": loss.item()})

        # save checkpoint each epoch
        ckpt_path = os.path.join(save_dir, f"unet_epoch{epoch+1}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"[INFO] Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    train()
