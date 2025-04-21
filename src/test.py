import glob
import os
import re

import cv2
import numpy as np
import torch
from config import CONFIG
from model import ColorizationModel


def get_best_model_path(model_folder):
    pattern = os.path.join(model_folder, "*best.pth")
    model_files = glob.glob(pattern)
    if not model_files:
        raise FileNotFoundError("No best model checkpoint found in the models folder.")

    def extract_epoch(filename):
        # Look for the pattern __<4-digit>__best in the filename
        match = re.search(r"__(\d{4})__best", filename)
        return int(match.group(1)) if match else 0

    # Sort files by extracted epoch number in descending order
    model_files = sorted(model_files, key=extract_epoch, reverse=True)
    return model_files[0]  # Return the most recent best model


def colorize_image(model, device, image_path, output_path):
    """
    Loads an image, resizes it for inference, then resizes the output back to the original size.
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
    L_resized = cv2.resize(
        L_channel,
        (CONFIG["data"]["crop"]["width"], CONFIG["data"]["crop"]["height"]),
        interpolation=cv2.INTER_LANCZOS4,
    )
    L_normalized = L_resized.astype(np.float32) / 255.0

    # Prepare tensor
    L_tensor = torch.from_numpy(L_normalized).unsqueeze(0).unsqueeze(0).to(device)

    # Model inference
    with torch.inference_mode():
        pred_AB = model(L_tensor)

    # Process model output
    pred_AB = pred_AB.squeeze().cpu().numpy()
    pred_AB = np.transpose(pred_AB, (1, 2, 0))
    pred_AB = pred_AB * 128.0 + 128.0

    # Combine L and predicted AB
    result_lab = np.concatenate([L_resized[:, :, np.newaxis], pred_AB], axis=2).astype(
        np.uint8
    )
    assert result_lab.shape == (
        CONFIG["data"]["crop"]["height"],
        CONFIG["data"]["crop"]["width"],
        3,
    )
    assert np.min(result_lab) >= 0 and np.max(result_lab) <= 255

    # Convert back to BGR
    result_bgr = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)

    # Resize back to original dimensions
    result_bgr = cv2.resize(
        result_bgr, (original_W, original_H), interpolation=cv2.INTER_LANCZOS4
    )

    # Save result
    cv2.imwrite(output_path, result_bgr)
    print(f"Colorized image saved to {output_path}")


def test():
    os.makedirs(CONFIG["testing"]["output_dir"], exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ColorizationModel(
        arch=CONFIG["model"]["arch"],
        encoder_name=CONFIG["model"]["encoder_name"],
        encoder_weights=CONFIG["model"]["encoder_weights"],
    ).to(device)

    # Load best model weights
    best_model_path = get_best_model_path(CONFIG["training"]["checkpoints_dir"])
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    print(f"Loaded best model from: {best_model_path}")

    # --------------------------
    # Process each image
    # --------------------------
    for file_name in os.listdir(CONFIG["data"]["test_dir"]):
        if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(CONFIG["data"]["test_dir"], file_name)
            output_path = os.path.join(CONFIG["testing"]["output_dir"], file_name)
            print(f"Processing {image_path}...")
            colorize_image(
                model,
                device,
                image_path,
                output_path,
            )


if __name__ == "__main__":
    test()
