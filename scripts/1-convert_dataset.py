import argparse
from gazetracker.dataset.kpadder import add_keypoints
from gazetracker.dataset.splitter import split_data, google_splitter, mit_splitter

"""
GOOGLE-SPLIT:
    Convert the Gaze Capture dataset (and prepare it for easy use in PyTorch) by following the pre-processing procedure
    that was used in Google research:
    "Accelerating eye movement research via accurate and affordable smartphone eye tracking", Nature communication 2020.
    Steps:
        1. Keep only portrait orientation images
        2. Keep only iPhone images
        3. Keep only images that have valid eye detections
        4. Split data of each participant into train, test, split

MIT-SPIT:
    Convert the Gaze Capture dataset (and prepare it for easy use in PyTorch) by following the pre-processing procedure
    that was used in the original paper announcing dataset release:
    "Eye Tracking for Everyone", CVPR 2016.
    Steps:
        1. Keep only portrait orientation images
        2. Create metadata file for each image
        3. Split data based on MIT annotation (separate participants in train, test, val)
"""


def parse_args():
    parser = argparse.ArgumentParser(description='Convert the MIT Gaze-Capture Dataset')
    parser.add_argument('--in_dir', type=str, default="../../dataset/",
                        help='Path to unzipped original MIT dataset.')
    parser.add_argument('--out_dir', type=str, default="../../google_split/",
                        help='Path to new converted dataset, that will have "image" and "meta" sub-folders inside each '
                             '"train", "val" and "test" folders.')
    parser.add_argument('--splitter', type=str, default="google-split",
                        help='The type of conversion process to use: can either be "google-split" or "mit-split".\n'
                             'Default is "google-split").')
    parser.add_argument('--predictor', default="./checkpoints/shape_predictor_68_face_landmarks.dat",
                        help='Path to trained facial landmark model file useful for adding key-points to new dataset.')
    parser.add_argument('--workers', type=int, default=1, help='Number of parallel threads.')
    return parser.parse_args()


def main():

    args = parse_args()

    print('Starting conversion...')
    if args.splitter == "google-split": splitter_function = google_splitter
    elif args.splitter == "mit-split": splitter_function = mit_splitter
    else: raise ValueError(f'The splitter parameter may either be "google-split" or "mit-split", not {args.splitter}.')
    split_data(in_dir=args.in_dir, out_dir=args.out_dir, splitter_function=splitter_function, workers=args.workers)
    print("Conversion Complete.")

    print('\nStarting to add key points...')
    add_keypoints(in_dir=args.out_dir, p=args.predictor)
    print("Key-points added.")


if __name__ == '__main__':
    main()
