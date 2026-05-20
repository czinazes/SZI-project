"""
camera.py — Webcam capture & real-time inference with overlay UI.

Improvements over v1:
  - Temporal smoothing: predictions averaged over last N frames (stops flickering)
  - Face padding: adds margin around detected face before cropping (matches training)
  - Probability bars: displays confidence bars for all emotion classes
  - Confidence threshold: shows "?" when model is uncertain
  - Histogram equalization: improves face quality in poor lighting
"""

import time
import collections
import cv2
import numpy as np

from src.config import cfg
from src.detect import get_face_detector, detect_faces
from src.model import load_trained_model

# ── Tunables ──────────────────────────────────────────────────────────────────
SMOOTH_FRAMES = 5          # number of frames to average predictions over
CONFIDENCE_THRESHOLD = 0.40  # below this → show "?" label
FACE_PADDING = 0.25        # fraction of face size to add as padding on each side
SHOW_BARS = True           # show probability bars for all emotions

# Emotion → BGR color mapping for bars
EMOTION_COLORS = {
    "Angry":    (0,   50,  220),
    "Disgust":  (0,   180, 80),
    "Fear":     (130, 0,   200),
    "Happy":    (0,   210, 255),
    "Sad":      (200, 100, 0),
    "Surprise": (0,   165, 255),
    "Neutral":  (160, 160, 160),
}
DEFAULT_COLOR = (200, 200, 200)
# ──────────────────────────────────────────────────────────────────────────────


def _get_emotion_color(label: str) -> tuple:
    return EMOTION_COLORS.get(label, DEFAULT_COLOR)


def extract_face_roi_padded(frame: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """
    Extract face ROI with extra padding on all sides.

    FER2013 images contain some forehead/chin context, so adding padding
    around the Haar detection box improves recognition quality.
    """
    pad_x = int(w * FACE_PADDING)
    pad_y = int(h * FACE_PADDING)

    fh, fw = frame.shape[:2]
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(fw, x + w + pad_x)
    y2 = min(fh, y + h + pad_y)

    face_roi = frame[y1:y2, x1:x2]

    # Convert to grayscale if needed
    if cfg.image.color_mode == "grayscale":
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        # Histogram equalization improves contrast in poor lighting
        face_roi = cv2.equalizeHist(face_roi)
    elif cfg.image.color_mode == "rgb":
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)

    face_roi = cv2.resize(face_roi, (cfg.image.size, cfg.image.size))
    face_roi = face_roi.astype("float32") / 255.0

    if cfg.image.channels == 1:
        face_roi = np.expand_dims(face_roi, axis=-1)

    return np.expand_dims(face_roi, axis=0)  # add batch dim


def draw_face_box(frame, x, y, w, h, label, confidence, color):
    """Draw bounding box and top label above the face."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, cfg.overlay.box_thickness)

    if confidence < CONFIDENCE_THRESHOLD:
        text = "?"
    elif cfg.overlay.show_confidence:
        text = f"{label}: {confidence:.0%}"
    else:
        text = label

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = cfg.overlay.font_scale
    thickness = 2

    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    # Background rectangle for text
    cv2.rectangle(frame, (x, y - th - 12), (x + tw + 8, y), color, -1)
    cv2.putText(frame, text, (x + 4, y - 5), font, scale, (255, 255, 255), thickness)


def draw_probability_bars(frame, x, y, w, all_probs: np.ndarray, class_names: list):
    """
    Draw a vertical bar chart of all emotion probabilities
    to the right of the face bounding box.
    """
    bar_x = x + w + 10
    bar_start_y = y
    bar_max_width = 120
    bar_height = 16
    bar_gap = 4
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.42

    fh, fw = frame.shape[:2]

    for i, (prob, name) in enumerate(zip(all_probs, class_names)):
        by = bar_start_y + i * (bar_height + bar_gap)
        if by + bar_height > fh or bar_x + bar_max_width > fw:
            break

        # Background
        cv2.rectangle(frame, (bar_x, by), (bar_x + bar_max_width, by + bar_height),
                      (50, 50, 50), -1)
        # Filled bar
        fill_w = int(bar_max_width * prob)
        bar_color = _get_emotion_color(name)
        cv2.rectangle(frame, (bar_x, by), (bar_x + fill_w, by + bar_height),
                      bar_color, -1)
        # Label + percentage
        label_text = f"{name[:3]} {prob:.0%}"
        cv2.putText(frame, label_text, (bar_x + 4, by + bar_height - 4),
                    font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)


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
    print(f"Smoothing: {SMOOTH_FRAMES} frames | Threshold: {CONFIDENCE_THRESHOLD:.0%} | Padding: {FACE_PADDING:.0%}")

    prev_time = time.time()

    # Per-face smoothing buffer: maps face_id → deque of prediction arrays
    # We use a simple single-buffer approach (one face assumed for now)
    smooth_buffer = collections.deque(maxlen=SMOOTH_FRAMES)

    # Session Statistics
    session_stats = {name: 0 for name in cfg.model.class_names}
    total_analyzed_faces = 0
    start_session_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame. Exiting.")
            break

        if cfg.camera.mirror:
            frame = cv2.flip(frame, 1)

        # FPS
        curr_time = time.time()
        fps = 1.0 / max(curr_time - prev_time, 1e-6)
        prev_time = curr_time

        # Detect faces
        faces = detect_faces(frame, detector)

        if len(faces) == 0:
            # No face → clear smoothing buffer
            smooth_buffer.clear()

        for i, (x, y, w, h) in enumerate(faces):
            # Padded ROI extraction with histogram equalization
            face_input = extract_face_roi_padded(frame, x, y, w, h)

            # Raw prediction
            raw_probs = model.predict(face_input, verbose=0)[0]

            # --- Temporal smoothing ---
            smooth_buffer.append(raw_probs)
            smoothed_probs = np.mean(smooth_buffer, axis=0)

            class_idx = int(np.argmax(smoothed_probs))
            confidence = float(smoothed_probs[class_idx])
            label = cfg.model.class_names[class_idx]

            # Choose box color
            if confidence < CONFIDENCE_THRESHOLD:
                color = (100, 100, 100)   # gray for uncertain
            else:
                color = _get_emotion_color(label)

            # Draw bounding box + top label
            draw_face_box(frame, x, y, w, h, label, confidence, color)
            
            # Update session stats
            session_stats[label] += 1
            total_analyzed_faces += 1

            # Draw probability bars for all emotions
            if SHOW_BARS:
                draw_probability_bars(frame, x, y, w, smoothed_probs,
                                      cfg.model.class_names)

        # FPS counter
        if cfg.overlay.show_fps:
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        # Smoothing info
        cv2.putText(frame, f"Smooth: {len(smooth_buffer)}/{SMOOTH_FRAMES}f",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow("Real-Time Emotion Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            filename = f"screenshot_{int(time.time())}.png"
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved: {filename}")

    cap.release()
    cv2.destroyAllWindows()
    
    # Print Session Statistics
    elapsed_time = time.time() - start_session_time
    print("\n" + "="*40)
    print("  📊 Session Statistics")
    print("="*40)
    print(f"Total Time:      {elapsed_time:.1f} seconds")
    print(f"Faces Analyzed:  {total_analyzed_faces}")
    
    if total_analyzed_faces > 0:
        print("\nEmotion Distribution:")
        sorted_stats = sorted(session_stats.items(), key=lambda x: x[1], reverse=True)
        for emotion, count in sorted_stats:
            if count > 0:
                pct = (count / total_analyzed_faces) * 100
                print(f"  - {emotion:10s}: {pct:5.1f}% ({count} frames)")
    print("="*40 + "\n")
    
    print("Camera closed.")
