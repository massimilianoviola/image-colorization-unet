import glob
import os
import re

import cv2
import numpy as np
import torch

from model import ColorizationModel


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

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Failed to load image at {image_path}")
        return

    # If the image is single-channel, convert it to 3 channels
    if len(img_bgr.shape) == 2 or img_bgr.shape[2] == 1:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)

    original_H, original_W = img_bgr.shape[:2]

    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L_channel = img_lab[:, :, 0].astype(np.float32)
    L_norm = L_channel / 255.0

    # Resize the L channel to target_size
    target_W, target_H = target_size
    resized_L = cv2.resize(L_norm, (target_W, target_H))

    L_tensor = torch.from_numpy(resized_L).unsqueeze(0).unsqueeze(0).to(device)

    with torch.inference_mode():
        pred_AB = model(L_tensor)  # shape: (1, 2, target_H, target_W)

    pred_AB_np = pred_AB.squeeze(0).cpu().numpy()  # shape: (2, target_H, target_W)

    # Reconstruct the LAB image at the target resolution:
    # Combine the resized L channel with predicted AB channels
    lab_result = np.concatenate(
        [resized_L[np.newaxis, :, :], pred_AB_np], axis=0
    )  # shape: (3, target_H, target_W)
    lab_result = np.transpose(lab_result, (1, 2, 0))  # shape: (target_H, target_W, 3)

    # Rescale from [0,1] to [0,255] for LAB representation
    lab_result_scaled = (lab_result * 255).astype(np.uint8)

    # Convert LAB to BGR and resize the result back to the original image dimensions
    result_bgr = cv2.cvtColor(lab_result_scaled, cv2.COLOR_LAB2BGR)
    result_bgr_resized = cv2.resize(result_bgr, (original_W, original_H))

    cv2.imwrite(output_path, result_bgr_resized)
    print(f"Colorized image saved to {output_path}")


def main():
    # --------------------------
    # Configuration
    # --------------------------
    model_folder = "./models"
    test_images_dir = "./test_images"
    output_folder = "./test_results"
    os.makedirs(output_folder, exist_ok=True)

    # Model parameters (must match those used during training)
    arch = "UnetPlusPlus"
    encoder_name = "efficientnet-b3"
    encoder_weights = "imagenet"
    target_size = (384, 384)

    # --------------------------
    # Automatically find the best model checkpoint
    # --------------------------
    best_model_path = get_best_model_path(model_folder)
    print(f"Best model checkpoint found: {best_model_path}")

    # --------------------------
    # Load Model
    # --------------------------
    model = ColorizationModel(
        arch=arch, encoder_name=encoder_name, encoder_weights=encoder_weights
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    print("Model loaded and set to evaluation mode.")

    # --------------------------
    # Infer each image in the test_images directory
    # --------------------------
    for file_name in os.listdir(test_images_dir):
        if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(test_images_dir, file_name)
            output_path = os.path.join(output_folder, file_name)
            print(f"Processing {image_path} ...")
            colorize_image(
                model, device, image_path, output_path, target_size=target_size
            )


if __name__ == "__main__":
    main()
