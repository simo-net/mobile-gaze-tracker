import os
import json
import torch
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Normalize, Resize, Compose, ToTensor  # , RandomCrop


class Gaze_Capture(Dataset):
    def __init__(self, root: str, split: str = 'train', size: (int, int) = (128, 128),
                 verbose: bool = True):

        self.files = glob(os.path.join(root, "images", "*.jpg"))
        # self.root = root
        # self.split = split
        # self.size = size
        # self.transform = transform
        self.resize_and_norm = self.get_transforms(size)
        # TODO: maye could add some data augmentation for train data

        if verbose:
            print(f"Number of images in the {split.upper()} set = {len(self.files)}")

    def __getitem__(self, idx: int):
        # Take image and metadata from file
        name = self.files[idx]
        image = Image.open(name)
        with open(name.replace('.jpg', '.json').replace('images', 'meta')) as f:
            meta = json.load(f)

        out = torch.tensor([meta['dot_x_cam'], meta['dot_y_cam']]).float()
        screen_w, screen_h = meta['screen_w'], meta['screen_h']

        # Key points
        w, h = image.size
        kps = [meta['leye_x1'] / w, meta['leye_y1'] / h, meta['leye_x2'] / w, meta['leye_y2'] / h,
               meta['reye_x1'] / w, meta['reye_y1'] / h, meta['reye_x2'] / w, meta['reye_y2'] / h]
        kps = torch.tensor(kps).float()

        # Right eye
        rx, ry, rw, rh = meta['reye_x'], meta['reye_y'], meta['reye_w'], meta['reye_h']
        r_eye = image.crop((max(0, rx), max(0, ry), max(0, rx + rw), max(0, ry + rh)))
        r_eye = self.resize_and_norm(r_eye)

        # Left eye
        lx, ly, lw, lh = meta['leye_x'], meta['leye_y'], meta['leye_w'], meta['leye_h']
        l_eye = image.crop((max(0, lx), max(0, ly), max(0, lx + lw), max(0, ly + lh)))
        l_eye = l_eye.transpose(Image.FLIP_LEFT_RIGHT)  # Note: we flip the left eye!
        l_eye = self.resize_and_norm(l_eye)

        return name, l_eye, r_eye, kps, out, screen_w, screen_h

    @staticmethod
    def get_transforms(size: (int, int)):
        list_transforms = [
            Resize((size[0], size[1])),
            ToTensor(),
            Normalize(mean=(0.3741, 0.4076, 0.5425), std=(0.02, 0.02, 0.02))  # TODO: check out mean and std of RGB eye-ROIs in the dataset
        ]
        return Compose(list_transforms)

    def __len__(self):
        return len(self.files)


# if __name__ == '__main__':
#     # Usage:
#     # >>    python3 loader.py  --root ~/gazecapture  --split 'test'
#     import argparse
#     from torch.utils.data import Dataset, DataLoader
#
#     parser = argparse.ArgumentParser(description='Load the Gaze-Capture pre-processed data.')
#     parser.add_argument('--root', type=str, default='~/gazecapture',
#                         help='Directory where dataset is stored.')
#     parser.add_argument('--split', type=str, default='test',
#                         help='Load either the train or test set.')
#     parser.add_argument('--batch_size', type=int, default=5,
#                         help='The batch size to use.')
#     parser.add_argument('--num_workers', type=int, default=6,
#                         help='The number of workers to use.')
#     parser.add_argument('--shuffle', action='store_true', default=False,
#                         help='Whether to shuffle the data.')
#     args = parser.parse_args()
#
#     dataset = Gaze_Capture(root=args.root+'/'+args.split, split=args.split)
#     loader = DataLoader(
#         dataset,
#         batch_size=args.batch_size,
#         num_workers=args.num_workers,
#         pin_memory=False,
#         shuffle=args.shuffle)
