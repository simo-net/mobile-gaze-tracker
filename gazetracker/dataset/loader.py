import os
import json
import torch
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Normalize, Resize, Compose, ToTensor  # , RandomCrop


class GazeCapture(Dataset):
    def __init__(self,
                 root: str or list,
                 size: (int, int) = (128, 128),
                 transforms = None,
                 verbose: bool = True):

        self.files = []
        if isinstance(root, list):
            self.split = []
            for directory in root:
                self.files += glob(os.path.join(directory, "images", "*.jpg"))
                self.split += [directory.split('/')[-1]]
        else:
            self.files = glob(os.path.join(root, "images", "*.jpg"))
        if not self.files:
            raise TypeError(f'No images found in {root}')
        self.files = sorted(self.files)

        self.size = size
        self.transforms = self.basic_transforms(size)
        if transforms is not None: self.transform_append(transforms)

        if verbose:
            split = self.files[0].split('/')[-3]
            print(f"Number of images in the {split.upper()} set = {len(self.files)}")

    def __getitem__(self, idx: int):
        # Take image and metadata from file
        name = self.files[idx]
        image = Image.open(name)
        with open(name.replace('.jpg', '.json').replace('images', 'meta')) as f:
            meta = json.load(f)

        out = torch.tensor([meta['dot_x_cam'], meta['dot_y_cam']]).float()
        # screen_w, screen_h = meta['screen_w'], meta['screen_h']

        # Key points
        w, h = image.size
        kps = [meta['leye_x1'] / w, meta['leye_y1'] / h, meta['leye_x2'] / w, meta['leye_y2'] / h,
               meta['reye_x1'] / w, meta['reye_y1'] / h, meta['reye_x2'] / w, meta['reye_y2'] / h]
        kps = torch.tensor(kps).float()

        # Right eye
        rx, ry, rw, rh = meta['reye_x'], meta['reye_y'], meta['reye_w'], meta['reye_h']
        r_eye = image.crop((max(0, rx), max(0, ry), max(0, rx + rw), max(0, ry + rh)))
        r_eye = self.transforms(r_eye)

        # Left eye
        lx, ly, lw, lh = meta['leye_x'], meta['leye_y'], meta['leye_w'], meta['leye_h']
        l_eye = image.crop((max(0, lx), max(0, ly), max(0, lx + lw), max(0, ly + lh)))
        l_eye = l_eye.transpose(Image.FLIP_LEFT_RIGHT)  # Note: we flip the left eye!
        l_eye = self.transforms(l_eye)

        return l_eye, r_eye, kps, out

    @staticmethod
    def basic_transforms(size: (int, int)):
        list_transforms = [
            Resize(size=size),
            ToTensor(),
            Normalize(mean=(0.3741, 0.4076, 0.5425), std=(0.02, 0.02, 0.02))  # TODO: check out mean and std of RGB eye-ROIs in the dataset
        ]
        return Compose(list_transforms)

    def transform_append(self, transform):
        if transform is None:
            return
        if self.transforms is None:
            self.transforms = transform
        else:
            self.transforms = Compose([self.transforms, transform])

    def __len__(self):
        return len(self.files)


def create_dataloader(
        dataset,
        batch_size = 72,
        shuffle: bool = True,
        seed: int = None,
        **dl_kwargs):
    if seed is not None:
        torch.random.manual_seed(seed)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **dl_kwargs)
    return dataloader


if __name__ == '__main__':
    # Usage:
    # >>    python3 loader.py  --root ~/gazecapture  --split 'test'
    import argparse

    parser = argparse.ArgumentParser(description='Load the Gaze-Capture pre-processed data.')
    parser.add_argument('--root', type=str, default='~/gazecapture',
                        help='Directory where dataset is stored.')
    parser.add_argument('--batch_size', type=int, default=5,
                        help='The batch size to use.')
    parser.add_argument('--shuffle', action='store_true', default=False,
                        help='Whether to shuffle the data.')
    args = parser.parse_args()

    data = GazeCapture(root=os.path.join(args.root, args.split), size=(128, 128), transforms=None, verbose=True)
    loader = create_dataloader(
        data,
        batch_size=args.batch_size,
        shuffle=args.shuffle, seed=None,
        num_workers=0, pin_memory=False
    )
