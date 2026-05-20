"""
detect.py — MediaPipe Face Detection utilities.
"""

import cv2
import numpy as np
import mediapipe as mp
from src.config import cfg


def get_face_detector():
    """Load MediaPipe Face Detection."""
    mp_face_detection = mp.solutions.face_detection
    return mp_face_detection.FaceDetection(
        model_selection=0, # 0 for short-range faces (within 2m)
        min_detection_confidence=0.5
    )


def detect_faces(frame: np.ndarray, detector) -> list:
    """
    Detect faces in a frame.

    Args:
        frame: BGR image from webcam.
        detector: MediaPipe FaceDetection instance.

    Returns:
        List of (x, y, w, h) tuples for each detected face.
    """
    # MediaPipe expects RGB images
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb_frame)
    
    faces = []
    if results.detections:
        fh, fw = frame.shape[:2]
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            x = int(bboxC.xmin * fw)
            y = int(bboxC.ymin * fh)
            w = int(bboxC.width * fw)
            h = int(bboxC.height * fh)
            
            # Ensure coordinates are within image boundaries
            x = max(0, x)
            y = max(0, y)
            w = min(w, fw - x)
            h = min(h, fh - y)
            
            if w > 0 and h > 0:
                faces.append((x, y, w, h))

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
