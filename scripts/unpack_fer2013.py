"""
unpack_fer2013.py — Unpack fer2013.csv into train/val/test directories with images.
"""

import os
import csv
import numpy as np
from PIL import Image

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

def main():
    csv_path = os.path.join("data", "processed", "fer2013.csv")
    output_dir = os.path.join("data", "processed")

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Please make sure the CSV file is placed there.")
        return

    # Ensure directories exist
    for usage in ["train", "val", "test"]:
        for emotion in EMOTIONS:
            os.makedirs(os.path.join(output_dir, usage, emotion), exist_ok=True)

    print("Unpacking fer2013.csv...")

    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)  # emotion, pixels, Usage
        
        rows = list(reader)
        total = len(rows)
        print(f"Total rows to process: {total}")
        
        iterator = tqdm(rows, desc="Unpacking images") if has_tqdm else rows
        
        for i, row in enumerate(iterator):
            if not has_tqdm and i % 5000 == 0:
                print(f"Processed {i}/{total} images...")
                
            emotion_id = int(row[0])
            pixels_str = row[1]
            usage = row[2]
            
            # Map usage
            if usage == "Training":
                usage_dir = "train"
            elif usage == "PublicTest":
                usage_dir = "val"
            elif usage == "PrivateTest":
                usage_dir = "test"
            else:
                continue
                
            emotion_name = EMOTIONS[emotion_id]
            
            # Convert pixels to image
            pixels = np.fromstring(pixels_str, dtype=np.uint8, sep=" ").reshape((48, 48))
            img = Image.fromarray(pixels)
            
            filename = f"{i}.png"
            save_path = os.path.join(output_dir, usage_dir, emotion_name, filename)
            img.save(save_path)

    print("Unpacking complete! All images saved to data/processed/")

if __name__ == "__main__":
    main()
