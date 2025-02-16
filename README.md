# image-colorization-unet

This project implements an image colorization model using U-Net++ architecture with EfficientNet backbone.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download training data:
```bash
# Download DIV2K dataset
./download_div2k_lr.sh

# Download BSD500 dataset (adds ~500 additional images)
./download_bsd500.sh
```

## Project Structure
```
.
├── config.yaml          # Configuration parameters
├── model.py            # U-Net++ model implementation
├── dataset.py          # Dataset loading and preprocessing
├── train.py            # Training script
├── test.py            # Testing/inference script
├── train_images/       # Training images (DIV2K + BSD500)
├── test_images/        # Test images
├── checkpoints/        # Saved model checkpoints
├── progress/           # Training progress visualizations
└── colorized_outputs/  # Colorized test outputs
```

## Training

1. Adjust parameters in config.yaml if needed
2. Run training:
```bash
python train.py
```

The training script will:
- Save model checkpoints to ./checkpoints/
- Save progress images to ./progress/
- Save the best model based on validation loss

## Testing

Run inference on test images:
```bash
python test.py
```

The test script will:
- Automatically find and load the best model
- Preserve original image resolution
- Save colorized outputs to ./colorized_outputs/

## Configuration

Key parameters in config.yaml:
```yaml
model:
  arch: "UnetPlusPlus"
  encoder_name: "efficientnet-b3"
  encoder_weights: "imagenet"

data:
  image_dir: "./test_images"
  crop:
    height: 384
    width: 384
  batch_size: 16
  num_workers: 4
```

Adjust these parameters to match your hardware capabilities and requirements.