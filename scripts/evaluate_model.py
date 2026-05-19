"""
evaluate_model.py — Evaluate the trained model on the test set.

Usage:
    python scripts/evaluate_model.py
    python scripts/evaluate_model.py --model models/best_model.keras
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluate import evaluate


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model file (default: from config.yaml)")
    args = parser.parse_args()

    evaluate(model_path=args.model)


if __name__ == "__main__":
    main()
