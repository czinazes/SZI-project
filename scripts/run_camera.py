"""
run_camera.py — Launch real-time webcam inference.

Usage:
    python scripts/run_camera.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.camera import run_camera


if __name__ == "__main__":
    run_camera()
