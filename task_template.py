import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import requests
from PIL import Image

# CONFIG
ZIP_FILE = "Dataset.zip"  # Path to the downloaded dataset zip
DATASET_DIR = Path("Dataset")  # Unzipped folder
TEMP_OUT_DIR = Path("submission_temp")  # Temporary folder for forged images
FILE_PATH = "submission.zip"  # Final file to upload

# Leaderboard submission
BASE_URL  = "http://35.192.205.84:80"
API_KEY  = "YOUR_API_KEY_HERE"  # REPLACE WITH YOUR API KEY
TASK_ID   = "22-forging-task"

# 1. UNZIP DATASET
if not DATASET_DIR.exists():
    if not os.path.exists(ZIP_FILE):
        raise FileNotFoundError(f"Could not find {ZIP_FILE}. Please download the dataset first.")

    print(f"Unzipping {ZIP_FILE}...")
    with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
        zip_ref.extractall(".")
else:
    print("Dataset already extracted.")

# Ensure output directory exists
TEMP_OUT_DIR.mkdir(exist_ok=True)


# 2. NAIVE FORGERY ATTACK (IMAGE AVERAGING)
print("Building forgery submission...")

# Map the Dataset structure: (Source_Folder, Size_Subfolder, Target_Folder)
CATEGORIES = [
    ("WM_1", 1, 25),
    ("WM_2", 26, 50),
    ("WM_3", 51, 75),
    ("WM_4", 76, 100),
    ("WM_5", 101, 125),
    ("WM_6", 126, 150),
    ("WM_7", 151, 175),
    ("WM_8", 176, 200),
]


total_processed = 0

for source_wm, target_start, target_stop in CATEGORIES:
    print(f"Processing {source_wm} dataset -> Forging onto images {target_start}.png to {target_stop}.png ...")

    source_dir = DATASET_DIR / "watermarked_sources" / source_wm
    source_images = list(source_dir.glob("*.png"))

    if not source_images:
        print(f"  [Warning] No source images found in {source_dir}")
        continue

    target_dir = DATASET_DIR / "clean_targets"
    target_images = []

    for number in range(target_start, target_stop + 1, 1):
        temp = target_dir / f"{number}.png"
        target_images.append(temp)

    for target_path, source_path in zip(target_images, source_images):
        # Load target clean image
        target_pil = Image.open(target_path).convert("RGB")

        # Load target source image
        source_pil = Image.open(source_path).convert("RGB")

        # Convert to numpy arrays for the math
        target_arr = np.array(target_pil).astype(np.float32)
        source_arr = np.array(source_pil).astype(np.float32)

        # Blend the Image with a Watermarked Image (Alpha Blending)
        forged_img = (target_arr * 0.5) + (source_arr * 0.5)

        # Clip values to valid pixel range [0, 255] and convert to uint8
        forged_img = np.clip(forged_img, 0, 255).astype(np.uint8)

        # Save to our temporary flat directory using the exact original filename (e.g., "104.png")
        out_path = TEMP_OUT_DIR / target_path.name
        Image.fromarray(forged_img).save(out_path)
        total_processed += 1

print(f"\nSuccessfully forged {total_processed} images.")
if total_processed != 200:
    print(f"[WARNING] Expected 200 images, but processed {total_processed}. Your submission may be rejected!")


# 3. PACKAGE INTO FLAT ZIP FILE
print(f"Packaging images into {FILE_PATH}...")
with zipfile.ZipFile(FILE_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
    for img_path in TEMP_OUT_DIR.glob("*.png"):
        zipf.write(img_path, arcname=img_path.name)

print(f"Saved submission file to {FILE_PATH}")


