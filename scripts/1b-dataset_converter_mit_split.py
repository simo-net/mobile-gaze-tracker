import argparse
from gazetracker.dataset.splitter import split_data, mit_splitter

"""
Convert the Gaze Capture dataset (and prepare it for easy use in PyTorch) by following the pre-processing procedure
that was used in the original paper announcing dataset release:
"Eye Tracking for Everyone", CVPR 2016.

Steps:
    1. Keep only portrait orientation images
    2. Create metadata file for each image
    3. Split data based on MIT annotation (separate participants in train, test, val)
"""

# Usage:
# python ./scripts/1b-dataset_converter_mit_split.py
#        --in_dir '/home/simonetunige/data/mobile-gaze/gaze-capture'
#        --out_dir '/home/simonetunige/data/mobile-gaze/gaze-capture_mit-split'
#        --workers 6


def main():
    parser = argparse.ArgumentParser(description='Convert the MIT Gaze-Capture Dataset')
    parser.add_argument('--in_dir', type=str, default="../../dataset/",
                        help='Path to unzipped MIT dataset.')
    parser.add_argument('--out_dir', type=str, default="../../mit_split/",
                        help='Path to new dataset should have "image" and "meta" folders with '
                             '"train", "val" and "test" subfolders.')
    parser.add_argument('--workers', type=int, default=1, help='Number of threads.')
    args = parser.parse_args()

    split_data(in_dir=args.in_dir, out_dir=args.out_dir, splitter_function=mit_splitter, workers=args.workers)
    print("Conversion Complete.")


if __name__ == "__main__":
    main()
