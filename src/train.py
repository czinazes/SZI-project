"""
train.py — Training loop with callbacks.

Handles the full training pipeline: data loading, model compilation,
training with early stopping and learning rate reduction, and saving
the best model.
"""

import os
from tensorflow import keras
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard,
)

from src.config import cfg
from src.data_preprocessing import create_train_generator
from src.model import build_model
from src.utils import plot_training_history


def get_callbacks() -> list:
    """
    Create a list of Keras callbacks for training.

    Returns:
        List of callback instances.
    """
    # Ensure model save directory exists
    os.makedirs(os.path.dirname(cfg.paths.model_save), exist_ok=True)

    callbacks = [
        # Save the best model based on validation accuracy
        ModelCheckpoint(
            filepath=cfg.paths.model_save,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        # Stop training if no improvement for N epochs
        EarlyStopping(
            monitor="val_accuracy",
            patience=cfg.training.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        # Reduce learning rate on plateau
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=cfg.training.reduce_lr_factor,
            patience=cfg.training.reduce_lr_patience,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    return callbacks


def train():
    """
    Execute the full training pipeline.

    Steps:
        1. Create data generators
        2. Build model
        3. Train with callbacks
        4. Plot training history
        5. Save results

    Returns:
        (model, history) — trained model and training history.
    """
    print("=" * 60)
    print("  Real-Time Face Analysis — Training Pipeline")
    print("=" * 60)
    print(f"  Task:           {cfg.task}")
    print(f"  Architecture:   {cfg.model.architecture}")
    print(f"  Classes:        {cfg.model.num_classes} ({', '.join(cfg.model.class_names)})")
    print(f"  Image size:     {cfg.image.size}x{cfg.image.size}x{cfg.image.channels}")
    print(f"  Epochs:         {cfg.training.epochs}")
    print(f"  Batch size:     {cfg.training.batch_size}")
    print(f"  Learning rate:  {cfg.training.learning_rate}")
    print("=" * 60)

    # Step 1: Create data generators
    print("\n[1/4] Loading and augmenting data...")
    train_gen, val_gen = create_train_generator()

    print(f"  Training samples:   {train_gen.samples}")
    print(f"  Validation samples: {val_gen.samples}")
    print(f"  Class indices:      {train_gen.class_indices}")

    # Step 2: Build model
    print("\n[2/4] Building model...")
    model = build_model()
    model.summary()

    # Step 3: Train
    print("\n[3/4] Training model...")
    callbacks = get_callbacks()

    history = model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // cfg.training.batch_size,
        validation_data=val_gen,
        validation_steps=val_gen.samples // cfg.training.batch_size,
        epochs=cfg.training.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Step 4: Plot results
    print("\n[4/4] Saving training history plot...")
    os.makedirs(os.path.dirname(cfg.paths.training_history_plot), exist_ok=True)
    plot_training_history(history, cfg.paths.training_history_plot)

    print(f"\nTraining complete!")
    print(f"  Best model saved to:  {cfg.paths.model_save}")
    print(f"  History plot saved to: {cfg.paths.training_history_plot}")

    return model, history
