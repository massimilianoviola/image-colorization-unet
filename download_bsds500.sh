#!/bin/bash
# Download the BSD500 dataset and extract into a training folder

# Directory to store the downloaded images
TRAIN_DIR="./train_images"
mkdir -p "$TRAIN_DIR"

# BSDS500 dataset URL from Berkeley
BSD_URL="https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"

# Create temp directory for extraction
TEMP_DIR="./temp_bsd"
mkdir -p "$TEMP_DIR"

echo "Downloading BSD500 dataset..."
wget -O "$TEMP_DIR/bsd500.tgz" "$BSD_URL"

echo "Extracting images..."
tar -xf "$TEMP_DIR/bsd500.tgz" -C "$TEMP_DIR"

# Move only image files from train, val, and test into TRAIN_DIR
echo "Moving image files to ${TRAIN_DIR}..."
find "$TEMP_DIR/BSR/BSDS500/data/images/train/" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) -exec mv {} "$TRAIN_DIR/" \;
find "$TEMP_DIR/BSR/BSDS500/data/images/val/" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) -exec mv {} "$TRAIN_DIR/" \;
find "$TEMP_DIR/BSR/BSDS500/data/images/test/" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) -exec mv {} "$TRAIN_DIR/" \;

echo "Cleaning up..."
rm -rf "$TEMP_DIR"

echo "Download complete. BSD500 images are now in ${TRAIN_DIR}"
