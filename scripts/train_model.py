"""
train_model.py — Train the CNN model.

Usage:
    python scripts/train_model.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import train


if __name__ == "__main__":
    train()
