"""
detect.py — Haar Cascade face detection utilities.
"""

import cv2
import numpy as np
from src.config import cfg


def get_face_detector() -> cv2.CascadeClassifier:
    """Load Haar Cascade face detector."""
    if cfg.camera.cascade_path:
        cascade_path = cfg.camera.cascade_path
    else:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError(f"Failed to load cascade from: {cascade_path}")

    return detector


def detect_faces(frame: np.ndarray, detector: cv2.CascadeClassifier) -> list:
    """
    Detect faces in a frame.

    Args:
        frame: BGR image from webcam.
        detector: Haar Cascade classifier.

    Returns:
        List of (x, y, w, h) tuples for each detected face.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=cfg.camera.detection_scale,
        minNeighbors=cfg.camera.min_neighbors,
        minSize=(cfg.camera.min_face_size, cfg.camera.min_face_size),
    )

    return faces


def extract_face_roi(frame: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """
    Extract and preprocess a face ROI for model input.

    Args:
        frame: BGR image.
        x, y, w, h: Face bounding box coordinates.

    Returns:
        Preprocessed face image ready for the CNN.
    """
    face_roi = frame[y:y + h, x:x + w]

    if cfg.image.color_mode == "grayscale":
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

    face_roi = cv2.resize(face_roi, (cfg.image.size, cfg.image.size))
    face_roi = face_roi.astype("float32") / 255.0

    if cfg.image.channels == 1:
        face_roi = np.expand_dims(face_roi, axis=-1)

    face_roi = np.expand_dims(face_roi, axis=0)  # batch dimension

    return face_roi
