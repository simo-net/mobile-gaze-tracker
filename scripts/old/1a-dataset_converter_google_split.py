import argparse
from gazetracker.dataset.splitter import split_data, google_splitter

"""
Convert the Gaze Capture dataset (and prepare it for easy use in PyTorch) by following the pre-processing procedure
that was used in Google research:
"Accelerating eye movement research via accurate and affordable smartphone eye tracking", Nature communication 2020.

Steps:
    1. Keep only portrait orientation images
    2. Keep only iPhone images
    3. Keep only images that have valid eye detections
    4. Split data of each participant into train, test, split
"""


def main():
    parser = argparse.ArgumentParser(description='Convert the MIT Gaze-Capture Dataset')
    parser.add_argument('--in_dir', type=str, default="../../dataset/",
                        help='Path to unzipped MIT dataset.')
    parser.add_argument('--out_dir', type=str, default="../../google_split/",
                        help='Path to new dataset should have "image" and "meta" sub-folders inside each '
                             '"train", "val" and "test" folders.')
    parser.add_argument('--workers', type=int, default=1, help='Number of threads')
    args = parser.parse_args()

    split_data(in_dir=args.in_dir, out_dir=args.out_dir, splitter_function=google_splitter, workers=args.workers)
    print("Conversion Complete.")


if __name__ == "__main__":
    main()
