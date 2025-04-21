import albumentations as A
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from config import CONFIG
from dataset import ImageColorizationDataset
from torch.utils.data import DataLoader


class ColorizationModel(nn.Module):
    def __init__(
        self,
        arch="Unet",
        encoder_name="efficientnet-b3",
        encoder_weights="imagenet",
        activation="tanh",
    ):
        """
        Initializes the colorization model using a U-Net architecture.
        Args:
            arch (str): The architecture to use (e.g., "Unet", "DeepLabV3+", etc.). The value is case-insensitive.
            encoder_name (str): Name of the encoder (from smp library).
            encoder_weights (str): Pretrained weights for the encoder.
            activation (str): Activation function to apply to the output. Using "tanh" will constrain the output
                to [-1, 1], which is desired for the scaled AB channels, so that gray is neutral 0.
        """
        super(ColorizationModel, self).__init__()
        self.model = smp.create_model(
            arch=arch,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=1,  # Input is L channel (grayscale)
            classes=2,  # Output is 2 AB channels
            activation=activation,
        )

    def forward(self, x):
        return self.model(x)


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ColorizationModel(
        arch=CONFIG["model"]["arch"],
        encoder_name=CONFIG["model"]["encoder_name"],
        encoder_weights=CONFIG["model"]["encoder_weights"],
    ).to(device)
    model.eval()

    with torch.inference_mode():
        for L, AB in dataloader:
            L = L.to(device)
            output = model(L)
            print("Input L channel shape:", L.shape)  # Expected: (batch_size, 1, H, W)
            print(
                "Predicted AB channels shape:", output.shape
            )  # Expected: (batch_size, 2, H, W)
            break
