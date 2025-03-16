#!/bin/bash
# Download the Stanford Background Dataset and extract into a training folder

# Directory to store the final images
TRAIN_DIR="./train_images"
mkdir -p "$TRAIN_DIR"

# Stanford Background Dataset URL
SBD_URL="http://dags.stanford.edu/data/iccv09Data.tar.gz"

# Create a temporary directory for downloading and extraction
TEMP_DIR="./temp_stanford"
mkdir -p "$TEMP_DIR"

echo "Downloading Stanford Background dataset..."
wget -O "$TEMP_DIR/iccv09Data.tar.gz" "$SBD_URL"

echo "Extracting dataset..."
tar -xzvf "$TEMP_DIR/iccv09Data.tar.gz" -C "$TEMP_DIR"

# Move only image files from the 'images' folder into TRAIN_DIR
echo "Moving image files to ${TRAIN_DIR}..."
find "$TEMP_DIR/iccv09Data/images/" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) -exec mv {} "$TRAIN_DIR/" \;

echo "Cleaning up temporary files..."
rm -rf "$TEMP_DIR"

echo "Download complete. Stanford Background dataset images are now in ${TRAIN_DIR}."
