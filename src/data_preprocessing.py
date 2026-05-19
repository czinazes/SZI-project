"""
data_preprocessing.py — Dataset loading, augmentation, and generator creation.

Uses tf.keras.preprocessing.image.ImageDataGenerator to create training,
validation, and test data pipelines with configurable augmentation.
"""

import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.config import cfg


def create_train_generator() -> tuple:
    """
    Create training and validation data generators with augmentation.

    Returns:
        (train_generator, val_generator)
    """
    # Training generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=cfg.augmentation.rotation_range,
        width_shift_range=cfg.augmentation.width_shift_range,
        height_shift_range=cfg.augmentation.height_shift_range,
        horizontal_flip=cfg.augmentation.horizontal_flip,
        zoom_range=cfg.augmentation.zoom_range,
        brightness_range=cfg.augmentation.brightness_range,
        validation_split=cfg.data.val_split,
    )

    target_size = (cfg.image.size, cfg.image.size)

    train_generator = train_datagen.flow_from_directory(
        cfg.data.train_dir,
        target_size=target_size,
        color_mode=cfg.image.color_mode,
        batch_size=cfg.training.batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
    )

    val_generator = train_datagen.flow_from_directory(
        cfg.data.train_dir,
        target_size=target_size,
        color_mode=cfg.image.color_mode,
        batch_size=cfg.training.batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
    )

    return train_generator, val_generator


def create_test_generator():
    """
    Create a test data generator (no augmentation, only rescaling).

    Returns:
        test_generator
    """
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    target_size = (cfg.image.size, cfg.image.size)

    test_generator = test_datagen.flow_from_directory(
        cfg.data.test_dir,
        target_size=target_size,
        color_mode=cfg.image.color_mode,
        batch_size=cfg.training.batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    return test_generator


def split_dataset(source_dir: str, output_dir: str, val_ratio: float = 0.2, test_ratio: float = 0.1):
    """
    Split a flat dataset directory into train/val/test.

    Expects source_dir to contain subdirectories for each class:
        source_dir/
            class_a/
                img1.jpg
                img2.jpg
            class_b/
                ...

    Creates:
        output_dir/
            train/class_a/  train/class_b/
            val/class_a/    val/class_b/
            test/class_a/   test/class_b/
    """
    classes = [d for d in os.listdir(source_dir)
               if os.path.isdir(os.path.join(source_dir, d))]

    for split in ["train", "val", "test"]:
        for cls in classes:
            os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

    for cls in classes:
        cls_dir = os.path.join(source_dir, cls)
        images = [f for f in os.listdir(cls_dir)
                  if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]

        # Split into train+val and test
        train_val, test = train_test_split(images, test_size=test_ratio, random_state=42)
        # Split train+val into train and val
        adjusted_val_ratio = val_ratio / (1 - test_ratio)
        train, val = train_test_split(train_val, test_size=adjusted_val_ratio, random_state=42)

        for img in train:
            shutil.copy2(os.path.join(cls_dir, img),
                         os.path.join(output_dir, "train", cls, img))
        for img in val:
            shutil.copy2(os.path.join(cls_dir, img),
                         os.path.join(output_dir, "val", cls, img))
        for img in test:
            shutil.copy2(os.path.join(cls_dir, img),
                         os.path.join(output_dir, "test", cls, img))

        print(f"  {cls}: {len(train)} train / {len(val)} val / {len(test)} test")

    print("Dataset split complete.")
