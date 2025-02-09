import os

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import ImageColorizationDataset
from model import ColorizationModel


def train():
    # --------------------------
    # Training configuration
    # --------------------------
    image_dir = "./train_images/DIV2K_train_LR_bicubic/X3"  # Directory containing your training images
    batch_size = 16  # Batch size for training
    num_epochs = 50  # Total number of training epochs
    learning_rate = 1e-3  # Learning rate for the optimizer
    arch = "UnetPlusPlus"  # Architecture to use
    encoder_name = "efficientnet-b3"  # Encoder backbone to use
    encoder_weights = "imagenet"  # Pretrained weights for the encoder

    # Folders for saving progress images and model checkpoints
    progress_folder = "./progress_images"
    model_folder = "./models"
    os.makedirs(progress_folder, exist_ok=True)
    os.makedirs(model_folder, exist_ok=True)

    # --------------------------
    # Define data augmentation / preprocessing
    # --------------------------
    # Since our input (L channel) is already scaled to [0, 1] and preprocessed as needed,
    # we simply apply augmentations and convert to a tensor.
    transform = A.Compose(
        [
            A.PadIfNeeded(
                min_height=384, min_width=384, border_mode=cv2.BORDER_REFLECT
            ),
            A.HorizontalFlip(p=0.5),
            A.RandomCrop(height=384, width=384),
            ToTensorV2(),
        ]
    )

    # --------------------------
    # Setup dataset and dataloader (Train/Val Split)
    # --------------------------
    dataset_full = ImageColorizationDataset(img_dir=image_dir, transform=transform)
    total_size = len(dataset_full)
    indices = list(range(total_size))
    np.random.shuffle(indices)
    split = int(np.floor(0.2 * total_size))  # 20% Val split
    val_indices = indices[:split]
    train_indices = indices[split:]
    train_dataset = Subset(dataset_full, train_indices)
    val_dataset = Subset(dataset_full, val_indices)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )  # shuffle to get diverse samples for visualization

    # --------------------------
    # Setup device and model
    # --------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ColorizationModel(
        arch=arch, encoder_name=encoder_name, encoder_weights=encoder_weights
    ).to(device)
    criterion = nn.L1Loss()
    optimizer = Adam(
        [
            {
                "params": model.model.encoder.parameters(),
                "lr": learning_rate / 100,
            },  # Lower lr for the pretrained encoder
            {
                "params": model.model.decoder.parameters(),
                "lr": learning_rate,
            },  # Higher lr for the decoder
        ]
    )
    best_val_loss = float("inf")  # Initialize best validation loss for checkpointing

    # --------------------------
    # Training loop
    # --------------------------
    print("Starting training...")
    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        train_progress = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} Training", leave=False
        )
        for L, AB in train_progress:
            L = L.to(device)
            AB = AB.to(device)

            preds = model(L)
            loss = criterion(preds, AB)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * L.size(0)
        avg_train_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_progress = tqdm(
            val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} Validation", leave=False
        )
        with torch.inference_mode():
            for L, AB in val_progress:
                L = L.to(device)
                AB = AB.to(device)
                preds = model(L)
                loss = criterion(preds, AB)
                val_loss += loss.item() * L.size(0)
        avg_val_loss = val_loss / len(val_dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Val Loss: {avg_val_loss:.4f}")

        # --------------------------
        # Save progress images at the end of each epoch
        # --------------------------
        model.eval()
        with torch.inference_mode():
            # Get one batch from the dataloader for visualization
            sample_L, sample_AB = next(iter(val_loader))
            sample_L = sample_L.to(device)
            sample_pred_AB = model(sample_L)

            # For visualization, take the first sample from the batch.
            L_np = sample_L[0].cpu().numpy()  # shape: (1, H, W)
            pred_AB_np = sample_pred_AB[0].cpu().numpy()  # shape: (2, H, W)
            gt_AB_np = sample_AB[0].cpu().numpy()  # shape: (2, H, W)

            # Combine L and AB channels to create a LAB image.
            # Note: Our dataset normalized LAB channels by dividing by 255,
            # so values are in [0, 1]. We convert them back to [0, 255] for visualization.
            lab_pred = np.concatenate([L_np, pred_AB_np], axis=0)  # shape: (3, H, W)
            lab_gt = np.concatenate([L_np, gt_AB_np], axis=0)

            # Rearrange to shape (H, W, 3)
            lab_pred = np.transpose(lab_pred, (1, 2, 0))
            lab_gt = np.transpose(lab_gt, (1, 2, 0))

            # Convert from [0, 1] to [0, 255]
            lab_pred = (lab_pred * 255).astype(np.uint8)
            lab_gt = (lab_gt * 255).astype(np.uint8)

            # Convert LAB images to BGR for saving using OpenCV
            bgr_pred = cv2.cvtColor(lab_pred, cv2.COLOR_LAB2BGR)
            bgr_gt = cv2.cvtColor(lab_gt, cv2.COLOR_LAB2BGR)

            # Save progress images into the progress folder
            cv2.imwrite(
                os.path.join(progress_folder, f"sample_epoch_{epoch + 1}_pred.jpg"),
                bgr_pred,
            )
            cv2.imwrite(
                os.path.join(progress_folder, f"sample_epoch_{epoch + 1}_gt.jpg"),
                bgr_gt,
            )
        # Save the model checkpoint if validation loss improved
        if avg_val_loss < best_val_loss:
            previous_best = best_val_loss
            best_val_loss = avg_val_loss
            model_filename = f"colorization_model__{arch}__{encoder_name}__{learning_rate}__{epoch + 1:04d}__best.pth"
            model_save_path = os.path.join(model_folder, model_filename)
            torch.save(model.state_dict(), model_save_path)
            print(
                f"Epoch {epoch + 1}: Validation loss improved from {previous_best:.4f} to {avg_val_loss:.4f}."
            )
            print(f"Best model updated and saved at: {model_save_path}")

    print("Training complete!")

    # --------------------------
    # Save the final model
    # --------------------------
    model_filename = f"colorization_model__{arch}__{encoder_name}__{learning_rate}__{num_epochs:04d}__last.pth"
    model_save_path = os.path.join(model_folder, model_filename)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model checkpoint saved as {model_save_path}")


if __name__ == "__main__":
    train()
