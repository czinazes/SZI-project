"""
model.py — CNN model definition.

Provides functions to build a custom CNN or a transfer-learning model
for face classification tasks.
"""

from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

from src.config import cfg


def build_mobilenet_transfer() -> keras.Model:
    """
    Build a Transfer Learning model using MobileNetV2.
    """
    input_shape = (cfg.image.size, cfg.image.size, cfg.image.channels)
    num_classes = cfg.model.num_classes

    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    
    # Freeze the base model
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])

    return model


def build_custom_cnn() -> keras.Model:
    """
    Build a custom Convolutional Neural Network.

    Architecture:
        4 × (Conv2D → BatchNorm → ReLU → MaxPool → Dropout)
        → Flatten → Dense(512) → Dense(256) → Softmax

    Returns:
        Compiled Keras model.
    """
    input_shape = (cfg.image.size, cfg.image.size, cfg.image.channels)
    num_classes = cfg.model.num_classes

    model = models.Sequential([
        # --- Block 1 ---
        layers.Conv2D(32, (3, 3), padding="same", input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        # --- Block 2 ---
        layers.Conv2D(64, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        # --- Block 3 ---
        layers.Conv2D(128, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        # --- Block 4 ---
        layers.Conv2D(256, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        # --- Fully Connected ---
        layers.Flatten(),

        layers.Dense(512),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.5),

        layers.Dense(256),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.5),

        # --- Output ---
        layers.Dense(num_classes, activation="softmax"),
    ])

    return model


def build_model() -> keras.Model:
    """
    Build and compile the model based on the configuration.

    Returns:
        Compiled Keras model ready for training.
    """
    if cfg.model.architecture == "custom_cnn":
        model = build_custom_cnn()
    elif cfg.model.architecture == "mobilenet_transfer":
        model = build_mobilenet_transfer()
    else:
        raise ValueError(f"Unknown architecture: {cfg.model.architecture}")

    # Compile
    optimizer = keras.optimizers.Adam(learning_rate=cfg.training.learning_rate)

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def load_trained_model(model_path: str = None) -> keras.Model:
    """
    Load a previously trained model from disk.

    Args:
        model_path: Path to the saved model. Defaults to config value.

    Returns:
        Loaded Keras model.
    """
    if model_path is None:
        model_path = cfg.paths.model_save

    model = keras.models.load_model(model_path)
    print(f"Model loaded from: {model_path}")
    return model
