"""
camera.py — Webcam capture & real-time inference with overlay UI.
"""

import time
import cv2
import numpy as np

from src.config import cfg
from src.detect import get_face_detector, detect_faces, extract_face_roi
from src.model import load_trained_model


def draw_overlay(frame, x, y, w, h, label, confidence, color):
    """Draw bounding box and label on frame."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, cfg.overlay.box_thickness)

    text = label
    if cfg.overlay.show_confidence:
        text = f"{label}: {confidence:.0%}"

    # Background for text
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                           cfg.overlay.font_scale, 2)
    cv2.rectangle(frame, (x, y - text_h - 10), (x + text_w + 5, y), color, -1)
    cv2.putText(frame, text, (x + 2, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                cfg.overlay.font_scale, (255, 255, 255), 2)


def run_camera():
    """Main loop: webcam → face detection → CNN prediction → overlay."""
    print("Loading model...")
    model = load_trained_model()

    print("Initializing face detector...")
    detector = get_face_detector()

    print(f"Opening webcam (index={cfg.camera.index})...")
    cap = cv2.VideoCapture(cfg.camera.index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.camera.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.camera.frame_height)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera index in config.yaml.")

    print("Press 'q' to quit, 's' to save screenshot.")
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame. Exiting.")
            break

        # FPS calculation
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        # Detect faces
        faces = detect_faces(frame, detector)

        for (x, y, w, h) in faces:
            # Extract and preprocess face ROI
            face_input = extract_face_roi(frame, x, y, w, h)

            # Predict
            predictions = model.predict(face_input, verbose=0)[0]
            class_idx = np.argmax(predictions)
            confidence = predictions[class_idx]
            label = cfg.model.class_names[class_idx]

            # Choose color based on confidence
            if confidence > 0.5:
                color = tuple(cfg.overlay.box_color_positive)
            else:
                color = tuple(cfg.overlay.box_color_negative)

            draw_overlay(frame, x, y, w, h, label, confidence, color)

        # Show FPS
        if cfg.overlay.show_fps:
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Real-Time Face Analysis", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            filename = f"screenshot_{int(time.time())}.png"
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved: {filename}")

    cap.release()
    cv2.destroyAllWindows()
    print("Camera closed.")
