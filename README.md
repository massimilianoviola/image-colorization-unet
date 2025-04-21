# 🎨 Image Colorization with U-Net

This project implements an image colorization model using the U-Net++ architecture with an EfficientNet-B3 backbone.
The model works in the LAB color space, where the L channel (grayscale) is used as input, and the model predicts the A and B color channels, which are then combined to reconstruct a full-color RGB image.

## 🔧 Setup

1. Install dependencies:
```bash
conda create -n image-colorization python==3.12 -y
conda activate image-colorization
pip install -r requirements.txt
```

2. Download training data:
```bash
# Download DIV2K train + validation (900 images), BSDS500 (500 images), and Stanford Background Dataset (715 images)
./data/download_data.sh
```

## 🏋️‍♂️ Training

1. Adjust parameters in `src/config.yaml` like batch size, learning rate, and epochs if needed.
2. Run training:
```bash
python src/train.py
```

The training script will:
- Save model checkpoints to `checkpoints/`
- Save progress images to `progress/`
- Save the best model based on validation loss

## 🧪 Testing

Run inference on test images in `./test_images/`:
```bash
python test.py
```

The test script will:
- Automatically find and load the best model
- Convert the image to grayscale and do inference, preserving the original image resolution
- Save colorized outputs to `./colorized_outputs/`

## ⚙️ Configuration

Key parameters in `config.yaml` that can be adjusted based on your hardware capabilities and requirements are:
```yaml
model:
  arch: UnetPlusPlus
  encoder_name: efficientnet-b3
  encoder_weights: imagenet

data:
  crop:
    height: 320
    width: 320
  batch_size: 24
  num_workers: 8

training:
  num_epochs: 50
  learning_rate: 0.0025
```
The code was tested on an NVIDIA Tesla T4 GPU.