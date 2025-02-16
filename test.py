import glob
import os
import re

import cv2
import numpy as np
import torch

from model import ColorizationModel
from config import CONFIG


def get_best_model_path(model_folder):
    pattern = os.path.join(model_folder, "*best.pth")
    model_files = glob.glob(pattern)
    if not model_files:
        raise FileNotFoundError("No best model checkpoint found in the models folder.")

    def extract_epoch(filename):
        match = re.search(r"epoch-(\d+)", filename)
        return int(match.group(1)) if match else 0

    model_files = sorted(model_files, key=extract_epoch, reverse=True)
    return model_files[0]


def colorize_image(model, device, image_path, output_path, target_size=(384, 384)):
    """
    Loads an image, resizes it to target_size for inference, then resizes the output back to the original size.
    The function extracts the L channel from the LAB representation, performs inference to predict AB channels,
    reconstructs the LAB image, converts it to BGR, and saves the final colorized image.
    """
    # Load and check image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Failed to load image at {image_path}")
        return

    # Store original dimensions
    original_H, original_W = img_bgr.shape[:2]

    # Convert to LAB color space
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L_channel = img_lab[:, :, 0]

    # Resize L channel to target size for model input
    L_resized = cv2.resize(L_channel, target_size)
    L_normalized = L_resized.astype(np.float32) / 255.0

    # Prepare tensor
    L_tensor = torch.from_numpy(L_normalized).unsqueeze(0).unsqueeze(0).to(device)

    # Model inference
    with torch.inference_mode():
        pred_AB = model(L_tensor)

    # Process model output
    pred_AB = pred_AB.squeeze().cpu().numpy()
    pred_AB = np.transpose(pred_AB, (1, 2, 0))
    
    # Combine L and predicted AB
    result_lab = np.concatenate([L_resized[:, :, np.newaxis], pred_AB * 255], axis=2)
    
    # Convert back to BGR
    result_bgr = cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    
    # Resize back to original dimensions
    result_bgr = cv2.resize(result_bgr, (original_W, original_H))
    
    # Save result
    cv2.imwrite(output_path, result_bgr)
    print(f"Colorized image saved to {output_path}")


def test():
    # --------------------------
    # Test configuration
    # --------------------------
    # Folders for loading model and saving colorized outputs
    os.makedirs(CONFIG['testing']['output_dir'], exist_ok=True)

    # --------------------------
    # Setup device and model
    # --------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ColorizationModel(
        arch=CONFIG['model']['arch'],
        encoder_name=CONFIG['model']['encoder_name'],
        encoder_weights=CONFIG['model']['encoder_weights']
    ).to(device)

    # Load best model weights
    best_model_path = get_best_model_path(CONFIG['training']['save_dir'])
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    print(f"Loaded best model from: {best_model_path}")

    # --------------------------
    # Process each image
    # --------------------------
    for file_name in os.listdir(CONFIG['data']['image_dir']):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(CONFIG['data']['image_dir'], file_name)
            output_path = os.path.join(CONFIG['testing']['output_dir'], file_name)
            print(f"Processing {image_path}...")
            colorize_image(
                model, 
                device, 
                image_path, 
                output_path, 
                target_size=(CONFIG['data']['crop']['width'], CONFIG['data']['crop']['height'])
            )


if __name__ == "__main__":
    test()
