import os

import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from config import CONFIG
from torch.utils.data import DataLoader, Dataset


class ImageColorizationDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        """
        Args:
            img_dir (str): Path to the directory with images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_dir = img_dir
        self.img_filenames = [
            os.path.join(self.img_dir, filename)
            for filename in os.listdir(self.img_dir)
            if filename.endswith((".png", ".jpg", ".jpeg"))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, idx):
        # Load the image as BGR
        img_path = self.img_filenames[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {img_path}")

        # Convert BGR to LAB
        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(
            "float32"
        )  # OpenCV outputs 8-bit LAB

        # Normalize LAB channels
        # - L: [0,255] → [0,1]
        # - A/B: [0,255] → [-1,1]
        lab_img[..., 0] = lab_img[..., 0] / 255.0
        lab_img[..., 1:] = (lab_img[..., 1:] - 128.0) / 128.0
        assert lab_img[..., 1:].min() >= -1.0 and lab_img[..., 1:].max() <= 1.0, (
            f"AB channels out of range: {lab_img[..., 1:].min()} to {lab_img[..., 1:].max()}"
        )

        # Apply transformations if provided
        if self.transform:
            lab_img = self.transform(image=lab_img)["image"]
        else:
            lab_img = torch.from_numpy(lab_img).permute(2, 0, 1)

        # Separate L and AB channels
        L_channel = lab_img[0:1, :, :]
        AB_channels = lab_img[1:, :, :]

        return L_channel, AB_channels


if __name__ == "__main__":
    transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomCrop(
                height=CONFIG["data"]["crop"]["height"],
                width=CONFIG["data"]["crop"]["width"],
            ),
            ToTensorV2(),
        ]
    )

    dataset = ImageColorizationDataset(
        img_dir=CONFIG["data"]["test_dir"], transform=transform
    )
    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG["data"]["batch_size"],
        num_workers=CONFIG["data"]["num_workers"],
        shuffle=True,
    )

    for L, AB in dataloader:
        print("L channel shape:", L.shape)  # Expected: (batch_size, 1, H, W)
        print("AB channels shape:", AB.shape)  # Expected: (batch_size, 2, H, W)
        break
