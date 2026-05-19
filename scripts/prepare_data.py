"""
prepare_data.py — Download & organize dataset.

Usage:
    python scripts/prepare_data.py --source data/raw/fer2013 --output data/processed
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import split_dataset


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for training")
    parser.add_argument("--source", type=str, required=True,
                        help="Source directory with class subdirectories")
    parser.add_argument("--output", type=str, default="data/processed",
                        help="Output directory for train/val/test splits")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="Validation split ratio (default: 0.2)")
    parser.add_argument("--test-ratio", type=float, default=0.1,
                        help="Test split ratio (default: 0.1)")
    args = parser.parse_args()

    print(f"Splitting dataset:")
    print(f"  Source:     {args.source}")
    print(f"  Output:     {args.output}")
    print(f"  Val ratio:  {args.val_ratio}")
    print(f"  Test ratio: {args.test_ratio}")

    split_dataset(args.source, args.output, args.val_ratio, args.test_ratio)


if __name__ == "__main__":
    main()
