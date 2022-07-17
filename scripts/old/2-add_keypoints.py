import argparse
from gazetracker.dataset.kpadder import add_keypoints


def main():
    parser = argparse.ArgumentParser(description='Adding eye key points to meta data')
    parser.add_argument('--dir', default='./dataset/', help='Path to converted dataset. Use dataset_converter_*.py')
    parser.add_argument('--predictor', default="./checkpoints/shape_predictor_68_face_landmarks.dat",
                        help='Path to trained facial landmark model file.')
    args = parser.parse_args()

    add_keypoints(in_dir=args.dir, p=args.predictor)
    print("Key-points added.")


if __name__ == '__main__':
    main()
