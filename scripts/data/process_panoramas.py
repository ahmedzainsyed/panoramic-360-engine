#!/usr/bin/env python3
"""Pre-process raw panoramas for training: normalize, resize, augment."""
import argparse
import os
import sys
import glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="datasets/raw", help="Input directory")
    parser.add_argument("--output", default="datasets/processed", help="Output directory")
    parser.add_argument("--target-width", type=int, default=2048)
    parser.add_argument("--augment", action="store_true")
    args = parser.parse_args()

    import cv2
    import numpy as np

    target_w = args.target_width
    target_h = target_w // 2

    for split in ["train", "val", "test"]:
        in_dir = os.path.join(args.input, split)
        out_img_dir = os.path.join(args.output, split, "images")
        out_mask_dir = os.path.join(args.output, split, "masks")
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_mask_dir, exist_ok=True)

        images = glob.glob(os.path.join(in_dir, "images", "*.jpg")) + \
                 glob.glob(os.path.join(in_dir, "images", "*.png"))

        for i, img_path in enumerate(images):
            img = cv2.imread(img_path)
            if img is None:
                continue
            # Resize to 2:1 equirectangular
            img_resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            out_name = os.path.splitext(os.path.basename(img_path))[0]
            cv2.imwrite(os.path.join(out_img_dir, f"{out_name}.jpg"), img_resized,
                       [cv2.IMWRITE_JPEG_QUALITY, 95])
            # Process corresponding mask if it exists
            mask_path = img_path.replace("/images/", "/masks/").replace(".jpg", ".png")
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask_resized = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(os.path.join(out_mask_dir, f"{out_name}.png"), mask_resized)

            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(images)} images for {split}")

    print("Data processing complete.")

if __name__ == "__main__":
    main()
