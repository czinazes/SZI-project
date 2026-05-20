"""
config.py — Configuration loader.

Reads config.yaml and provides a global Config object accessible throughout
the project.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Resolve project root (one level up from src/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")


def _load_yaml() -> dict:
    """Load the YAML configuration file."""
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Typed dataclasses for each config section
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    raw_dir: str
    processed_dir: str
    train_dir: str
    val_dir: str
    test_dir: str
    val_split: float = 0.2


@dataclass
class ImageConfig:
    size: int = 48
    channels: int = 1
    color_mode: str = "grayscale"


@dataclass
class ModelConfig:
    architecture: str = "custom_cnn"
    num_classes: int = 7
    class_names: List[str] = field(default_factory=lambda: [
        "Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"
    ])


@dataclass
class TrainingConfig:
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    optimizer: str = "adam"
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5


@dataclass
class AugmentationConfig:
    rotation_range: int = 15
    width_shift_range: float = 0.1
    height_shift_range: float = 0.1
    horizontal_flip: bool = True
    zoom_range: float = 0.1
    brightness_range: Tuple[float, float] = (0.9, 1.1)


@dataclass
class PathsConfig:
    model_save: str = "models/best_model.keras"
    training_history_plot: str = "outputs/training_history.png"
    confusion_matrix_plot: str = "outputs/confusion_matrix.png"


@dataclass
class CameraConfig:
    index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    cascade_path: Optional[str] = None
    min_face_size: int = 30
    detection_scale: float = 1.3
    min_neighbors: int = 5
    mirror: bool = False


@dataclass
class OverlayConfig:
    box_color_positive: List[int] = field(default_factory=lambda: [0, 255, 0])
    box_color_negative: List[int] = field(default_factory=lambda: [0, 0, 255])
    box_thickness: int = 2
    font_scale: float = 0.8
    show_confidence: bool = True
    show_fps: bool = True


@dataclass
class Config:
    task: str
    data: DataConfig
    image: ImageConfig
    model: ModelConfig
    training: TrainingConfig
    augmentation: AugmentationConfig
    paths: PathsConfig
    camera: CameraConfig
    overlay: OverlayConfig


# ---------------------------------------------------------------------------
# Build Config from YAML
# ---------------------------------------------------------------------------

def load_config() -> Config:
    """Parse config.yaml into a typed Config object."""
    raw = _load_yaml()

    # Resolve all relative paths to absolute
    def _abs(path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.join(PROJECT_ROOT, path)

    data_raw = raw.get("data", {})
    data_cfg = DataConfig(
        raw_dir=_abs(data_raw.get("raw_dir", "data/raw")),
        processed_dir=_abs(data_raw.get("processed_dir", "data/processed")),
        train_dir=_abs(data_raw.get("train_dir", "data/processed/train")),
        val_dir=_abs(data_raw.get("val_dir", "data/processed/val")),
        test_dir=_abs(data_raw.get("test_dir", "data/processed/test")),
        val_split=data_raw.get("val_split", 0.2),
    )

    img_raw = raw.get("image", {})
    image_cfg = ImageConfig(
        size=img_raw.get("size", 48),
        channels=img_raw.get("channels", 1),
        color_mode=img_raw.get("color_mode", "grayscale"),
    )

    mdl_raw = raw.get("model", {})
    model_cfg = ModelConfig(
        architecture=mdl_raw.get("architecture", "custom_cnn"),
        num_classes=mdl_raw.get("num_classes", 7),
        class_names=mdl_raw.get("class_names", []),
    )

    trn_raw = raw.get("training", {})
    training_cfg = TrainingConfig(
        epochs=trn_raw.get("epochs", 50),
        batch_size=trn_raw.get("batch_size", 32),
        learning_rate=trn_raw.get("learning_rate", 0.001),
        optimizer=trn_raw.get("optimizer", "adam"),
        early_stopping_patience=trn_raw.get("early_stopping_patience", 10),
        reduce_lr_patience=trn_raw.get("reduce_lr_patience", 5),
        reduce_lr_factor=trn_raw.get("reduce_lr_factor", 0.5),
    )

    aug_raw = raw.get("augmentation", {})
    augmentation_cfg = AugmentationConfig(
        rotation_range=aug_raw.get("rotation_range", 15),
        width_shift_range=aug_raw.get("width_shift_range", 0.1),
        height_shift_range=aug_raw.get("height_shift_range", 0.1),
        horizontal_flip=aug_raw.get("horizontal_flip", True),
        zoom_range=aug_raw.get("zoom_range", 0.1),
        brightness_range=tuple(aug_raw.get("brightness_range", [0.9, 1.1])),
    )

    pth_raw = raw.get("paths", {})
    paths_cfg = PathsConfig(
        model_save=_abs(pth_raw.get("model_save", "models/best_model.keras")),
        training_history_plot=_abs(pth_raw.get("training_history_plot", "outputs/training_history.png")),
        confusion_matrix_plot=_abs(pth_raw.get("confusion_matrix_plot", "outputs/confusion_matrix.png")),
    )

    cam_raw = raw.get("camera", {})
    cascade = cam_raw.get("cascade_path")
    if cascade is not None:
        cascade = _abs(cascade)
    camera_cfg = CameraConfig(
        index=cam_raw.get("index", 0),
        frame_width=cam_raw.get("frame_width", 640),
        frame_height=cam_raw.get("frame_height", 480),
        cascade_path=cascade,
        min_face_size=cam_raw.get("min_face_size", 30),
        detection_scale=cam_raw.get("detection_scale", 1.3),
        min_neighbors=cam_raw.get("min_neighbors", 5),
        mirror=cam_raw.get("mirror", False),
    )

    ovl_raw = raw.get("overlay", {})
    overlay_cfg = OverlayConfig(
        box_color_positive=ovl_raw.get("box_color_positive", [0, 255, 0]),
        box_color_negative=ovl_raw.get("box_color_negative", [0, 0, 255]),
        box_thickness=ovl_raw.get("box_thickness", 2),
        font_scale=ovl_raw.get("font_scale", 0.8),
        show_confidence=ovl_raw.get("show_confidence", True),
        show_fps=ovl_raw.get("show_fps", True),
    )

    return Config(
        task=raw.get("task", "emotion"),
        data=data_cfg,
        image=image_cfg,
        model=model_cfg,
        training=training_cfg,
        augmentation=augmentation_cfg,
        paths=paths_cfg,
        camera=camera_cfg,
        overlay=overlay_cfg,
    )


# Singleton — import and use: `from src.config import cfg`
cfg = load_config()
