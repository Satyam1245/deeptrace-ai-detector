#!/usr/bin/env python3
"""
Dataset augmentation utility for deepfake image training sets.
Creates augmented copies of images from real/fake source folders.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
from albumentations import (
    Compose,
    HorizontalFlip,
    RandomResizedCrop,
    Rotate,
    ShiftScaleRotate,
    RandomBrightnessContrast,
    GaussNoise,
    ImageCompression,
    Blur,
    Resize,
)


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(paths):
    files = []
    for folder in paths:
        base = Path(folder)
        if not base.exists():
            print(f"Skipping missing folder: {base}")
            continue
        for item in base.rglob("*"):
            if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(item)
    return files


def augment_class(image_paths, output_dir, copies_per_image, image_size):
    output_dir.mkdir(parents=True, exist_ok=True)
    transform = Compose([
        HorizontalFlip(p=0.5),
        RandomResizedCrop(image_size, image_size, scale=(0.75, 1.0), p=0.5),
        Rotate(limit=12, p=0.3),
        ShiftScaleRotate(shift_limit=0.05, scale_limit=0.08, rotate_limit=8, p=0.3),
        RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
        GaussNoise(var_limit=(5.0, 20.0), p=0.2),
        ImageCompression(quality_lower=80, quality_upper=100, p=0.2),
        Blur(blur_limit=3, p=0.15),
        Resize(image_size, image_size),
    ])

    for image_path in image_paths:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        stem = image_path.stem
        suffix = image_path.suffix.lower()

        original_output = output_dir / f"{stem}_orig{suffix}"
        cv2.imwrite(str(original_output), image_bgr)

        for idx in range(copies_per_image):
            augmented = transform(image=image_rgb)["image"]
            augmented_bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
            augmented_path = output_dir / f"{stem}_aug_{idx + 1}{suffix}"
            cv2.imwrite(str(augmented_path), augmented_bgr)


def main():
    parser = argparse.ArgumentParser(description="Augment real/fake image datasets")
    parser.add_argument("--real-dirs", nargs="*", default=[], help="Source folders with real images")
    parser.add_argument("--fake-dirs", nargs="*", default=[], help="Source folders with fake images")
    parser.add_argument("--output-dir", default="outputs/augmented_dataset", help="Where augmented images will be saved")
    parser.add_argument("--copies-per-image", type=int, default=3, help="Number of augmented copies to create")
    parser.add_argument("--image-size", type=int, default=240, help="Augmentation image size")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    real_images = collect_images(args.real_dirs)
    fake_images = collect_images(args.fake_dirs)

    if not real_images and not fake_images:
        print("No source images found. Provide --real-dirs and/or --fake-dirs.")
        return

    if real_images:
        augment_class(real_images, output_dir / "real", args.copies_per_image, args.image_size)
        print(f"Augmented {len(real_images)} real images into {output_dir / 'real'}")

    if fake_images:
        augment_class(fake_images, output_dir / "fake", args.copies_per_image, args.image_size)
        print(f"Augmented {len(fake_images)} fake images into {output_dir / 'fake'}")


if __name__ == "__main__":
    main()
