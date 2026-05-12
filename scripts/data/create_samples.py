#!/usr/bin/env python3
"""Create minimal sample datasets for testing."""
import numpy as np
import cv2
import os

def create_sample_panorama(output_path: str, width: int = 512, height: int = 256):
    """Create a synthetic construction-like panorama."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # Sky (top 40%)
    sky_h = int(height * 0.4)
    img[:sky_h, :] = [100, 140, 200]
    # Ground (bottom 35%)
    ground_start = int(height * 0.65)
    img[ground_start:, :] = [80, 65, 50]
    # Structures (middle)
    img[sky_h:ground_start, :] = [120, 110, 100]
    # Wall
    img[sky_h:ground_start, 100:300] = [160, 160, 155]
    # Scaffolding
    for x in range(110, 290, 20):
        img[sky_h:ground_start, x:x+3] = [200, 130, 50]
    # Worker (blue dot)
    cv2.circle(img, (200, int(height * 0.7)), 15, [0, 0, 180], -1)
    cv2.circle(img, (200, int(height * 0.63)), 8, [200, 150, 100], -1)
    # Add slight noise
    noise = np.random.normal(0, 8, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"Created panorama: {output_path} ({width}x{height})")
    return img

def create_sample_mask(output_path: str, width: int = 512, height: int = 256):
    """Create a synthetic segmentation mask matching the sample panorama."""
    mask = np.zeros((height, width), dtype=np.uint8)
    sky_h = int(height * 0.4)
    mask[:sky_h, :] = 1             # sky
    mask[int(height*0.65):, :] = 2  # ground
    mask[sky_h:int(height*0.65), :] = 3  # wall
    mask[sky_h:int(height*0.65), 100:300] = 7  # concrete
    for x in range(110, 290, 20):
        mask[sky_h:int(height*0.65), x:x+3] = 10  # scaffolding
    # Worker
    cy, cx = int(height*0.7), 200
    for dy in range(-15, 16):
        for dx in range(-15, 16):
            if dx*dx + dy*dy <= 225 and 0 <= cy+dy < height and 0 <= cx+dx < width:
                mask[cy+dy, cx+dx] = 14  # worker
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    cv2.imwrite(output_path, mask)
    print(f"Created mask: {output_path}")
    return mask

if __name__ == "__main__":
    create_sample_panorama("datasets/samples/panorama_sample.jpg")
    create_sample_mask("datasets/samples/annotation_sample.png")
    print("\nSample dataset created successfully!")
