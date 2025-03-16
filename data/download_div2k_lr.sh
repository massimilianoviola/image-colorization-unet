#!/bin/bash
# Download the DIV2K lower-resolution (bicubic) training and validation images and extract them into a training folder

# Directory to store the downloaded images
TRAIN_DIR="./train_images"
mkdir -p "$TRAIN_DIR"

# URL for the DIV2K lower resolution images (bicubic). More at https://data.vision.ee.ethz.ch/cvl/DIV2K/
TRAIN_URL="https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X3.zip"
VAL_URL="https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X3.zip"

# Output zip filenames
TRAIN_ZIP="DIV2K_train_LR_bicubic.zip"
VAL_ZIP="DIV2K_valid_LR_bicubic.zip"

echo "Downloading DIV2K training images..."
wget -O "$TRAIN_ZIP" "$TRAIN_URL"

echo "Downloading DIV2K validation images..."
wget -O "$VAL_ZIP" "$VAL_URL"

echo "Extracting images to ${TRAIN_DIR}..."
unzip -j "$TRAIN_ZIP" -d "$TRAIN_DIR"
unzip -j "$VAL_ZIP" -d "$TRAIN_DIR"

echo "Extraction complete. Deleting the ZIP files..."
rm "$TRAIN_ZIP" "$VAL_ZIP"

echo "Download, extraction, and cleanup complete. The DIV2K lower-resolution images are now in ${TRAIN_DIR}."
