#!/bin/bash
# Download the DIV2K lower-resolution (bicubic) training images and extract them into a training folder

# Directory to store the downloaded images
TRAIN_DIR="./train_images"
mkdir -p "$TRAIN_DIR"

# URL for the DIV2K lower resolution images (bicubic). More at https://data.vision.ee.ethz.ch/cvl/DIV2K/
DIV2K_LR_URL="https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X3.zip"

# Output zip filename
ZIP_FILE="DIV2K_train_LR_bicubic.zip"

echo "Downloading DIV2K lower-resolution images..."
wget -O "$ZIP_FILE" "$DIV2K_LR_URL"

echo "Extracting images to ${TRAIN_DIR}..."
unzip "$ZIP_FILE" -d "$TRAIN_DIR"

echo "Extraction complete. Deleting the ZIP file..."
rm "$ZIP_FILE"

echo "Download, extraction, and cleanup complete. The DIV2K lower-resolution images are now in ${TRAIN_DIR}."
