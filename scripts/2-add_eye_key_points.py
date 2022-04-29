import argparse
from gaze_tracker.dataset.kpadder import add_keypoints


def main():
    parser = argparse.ArgumentParser(description='Adding eye key points to meta data')
    parser.add_argument('--dir', default='./dataset/', help='Path to converted dataset. Use dataset_converter')
    parser.add_argument('--p', default="./checkpoints/shape_predictor_68_face_landmarks.dat",
                        help='Path to trained facial landmark model file')
    parser.add_argument('--workers', default=40, type=int, help='Number of CPU cores available')
    args = parser.parse_args()

    add_keypoints(in_dir=args.dir, p=args.p, workers=args.workers)
    print("Key-points added.")


if __name__ == '__main__':
    main()
