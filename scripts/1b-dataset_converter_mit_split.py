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


def main():
    parser = argparse.ArgumentParser(description='Convert the MIT Gaze-Capture Dataset')
    parser.add_argument('--dir', default="../../dataset/", help='Path to unzipped MIT dataset')
    parser.add_argument('--out_dir', default="../../mit_split/",
                        help='Path to new dataset should have image, meta folders with train, val, test subfolders')
    parser.add_argument('--workers', default=1, help='Number of threads', type=int)
    args = parser.parse_args()

    split_data(in_dir=args.dir, out_dir=args.out_dir, splitter_function=mit_splitter, workers=args.workers)
    print("Conversion Complete.")


if __name__ == "__main__":
    main()
