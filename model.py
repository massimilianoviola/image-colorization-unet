import albumentations as A
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from dataset import ImageColorizationDataset


class ColorizationModel(nn.Module):
    def __init__(
        self,
        arch="UnetPlusPlus",
        encoder_name="efficientnet-b3",
        encoder_weights="imagenet",
        activation="sigmoid",
    ):
        """
        Initializes the colorization model using a U-Net architecture.
        Args:
            arch (str): The architecture to use (e.g., "Unet", "DeepLabV3+", etc.). The value is case-insensitive.
            encoder_name (str): Name of the encoder (from smp library).
            encoder_weights (str): Pretrained weights for the encoder.
            activation (str): Activation function to apply to the output. Using "sigmoid" will constrain the output
                to [0, 1], which is desired for the scaled AB channels.
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
    image_dir = "./test_images"
    arch = "UnetPlusPlus"
    encoder_name = "efficientnet-b3"
    encoder_weights = "imagenet"

    transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomCrop(height=384, width=384),
            ToTensorV2(),
        ]
    )

    dataset = ImageColorizationDataset(img_dir=image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ColorizationModel(
        arch=arch, encoder_name=encoder_name, encoder_weights=encoder_weights
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
