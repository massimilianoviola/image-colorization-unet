import os

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
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
            os.path.join(img_dir, fname) for fname in os.listdir(img_dir) if fname.endswith((".png", ".jpg", ".jpeg"))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, idx):
        # Load the image as BGR
        img_path = self.img_filenames[idx]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image not found at {img_path}")

        # Convert BGR to LAB
        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype("float32")  # OpenCV outputs 8-bit LAB

        # Normalize LAB channels [0, 255] -> [0, 1]
        lab_img /= 255.0

        # Apply transformations if provided
        if self.transform:
            lab_img = self.transform(image=lab_img)["image"]

        # Separate L and AB channels
        L_channel = lab_img[0:1, :, :]
        AB_channels = lab_img[1:, :, :]

        return L_channel, AB_channels


if __name__ == "__main__":
    image_dir = "./test_images"

    transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomCrop(height=256, width=256),
            ToTensorV2(),
        ]
    )

    dataset = ImageColorizationDataset(img_dir=image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    for L, AB in dataloader:
        print("L channel shape:", L.shape)  # (batch_size, 1, H, W)
        print("AB channels shape:", AB.shape)  # (batch_size, 2, H, W)
        break
