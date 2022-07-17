import os
import json
import numpy as np
from glob import glob
from PIL import Image
import tensorflow as tf
from skimage.transform import resize


class GazeCapture:
    def __init__(self,
                 root: str or list,
                 size: (int, int) = (128, 128),
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

        if verbose:
            split = self.files[0].split('/')[-3]
            print(f"Number of images in the {split.upper()} set = {len(self.files)}")

    def __getitem__(self, idx: int):
        # Take image and metadata from file
        name = self.files[idx]
        image = Image.open(name)
        with open(name.replace('.jpg', '.json').replace('images', 'meta')) as f:
            meta = json.load(f)

        out = np.array([meta['dot_x_cam'], meta['dot_y_cam']], dtype=float)
        # screen_w, screen_h = meta['screen_w'], meta['screen_h']

        # Key points
        w, h = image.size
        kps = [meta['leye_x1'] / w, meta['leye_y1'] / h, meta['leye_x2'] / w, meta['leye_y2'] / h,
               meta['reye_x1'] / w, meta['reye_y1'] / h, meta['reye_x2'] / w, meta['reye_y2'] / h]
        kps = np.array(kps, dtype=float)

        # Right eye
        rx, ry, rw, rh = meta['reye_x'], meta['reye_y'], meta['reye_w'], meta['reye_h']
        r_eye = image.crop((max(0, rx), max(0, ry), max(0, rx + rw), max(0, ry + rh)))
        r_eye = self.transforms(r_eye)

        # Left eye
        lx, ly, lw, lh = meta['leye_x'], meta['leye_y'], meta['leye_w'], meta['leye_h']
        l_eye = image.crop((max(0, lx), max(0, ly), max(0, lx + lw), max(0, ly + lh)))
        l_eye = l_eye.transpose(Image.FLIP_LEFT_RIGHT)  # Note: we flip the left eye!
        l_eye = self.transforms(l_eye)

        r_eye, l_eye = tf.cast(r_eye, tf.float32), tf.cast(l_eye, tf.float32)
        kps, out = tf.cast(kps, tf.float32), tf.cast(out, tf.float32)
        return l_eye, r_eye, kps, out

    def transforms(self, img):
        img = np.array(img)
        img = Resize(
            img,
            size=self.size
        )
        img = Normalize(
            img,
            mean=(0.3741, 0.4076, 0.5425),
            std=(0.02, 0.02, 0.02)
        )  # TODO: check out mean and std of RGB eye-ROIs in the dataset
        return img

    def __len__(self):
        return len(self.files)


def Normalize(image: np.array, mean: list or tuple, std: list or tuple):
    image /= 255
    for channel in range(3):
        image[:,:,channel] = (image[:,:,channel] - mean[channel]) / std[channel]
    return image


def Resize(image: np.array, size: list or tuple):
    image = resize(image, output_shape=size, order=1)
    # image = tf.image.resize(image, size=size)  # TODO: add resizing function, should work with numpy array instead than tf tensor
    return image


def shuffle_data(dataset, seed: int = None):
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def create_data_generator(dataset,
                          ids: np.array = None,
                          shuffle: bool = False, seed: int = None):
    while True:
        ids = ids if ids is not None and len(ids) > 0 else range(0, len(dataset))
        if shuffle:
            shuffle_data(ids, seed=seed)
        for i in ids:
            l_eye, r_eye, kps, y = dataset[i]
            yield l_eye, r_eye, kps, y


def create_dataloader(dataset,
                      batch_size: int = 72,
                      shuffle: bool = False, seed: int = None):
    while True:
        batch_ids = np.arange(0, len(dataset), batch_size, dtype=int)
        if shuffle:
            shuffle_data(batch_ids, seed=seed)
        for bi in batch_ids:
            l_eye_batch, r_eye_batch = [], []
            kps_batch, y_batch = [], []
            for i in range(bi, bi+batch_size):
                l_eye, r_eye, kps, y = dataset[i]
                l_eye_batch.append(l_eye)
                r_eye_batch.append(r_eye)
                kps_batch.append(kps)
                y_batch.append(y)
            l_eye_batch = tf.convert_to_tensor(l_eye_batch, dtype='float32')
            r_eye_batch = tf.convert_to_tensor(r_eye_batch, dtype='float32')
            kps_batch = tf.convert_to_tensor(kps_batch, dtype='float32')
            y_batch = tf.convert_to_tensor(y_batch, dtype='float32')
            yield l_eye_batch, r_eye_batch, kps_batch, y_batch


if __name__ == '__main__':
    # Usage:
    # >>    python3 loader.py  --root ~/gazecapture  --split 'test'
    import argparse

    parser = argparse.ArgumentParser(description='Load the Gaze-Capture pre-processed data.')
    parser.add_argument('--root', type=str, default='~/gazecapture',
                        help='Directory where dataset is stored.')
    parser.add_argument('--split', type=str, default='test',
                        help='Load either the train or test set.')
    parser.add_argument('--batch_size', type=int, default=5,
                        help='The batch size to use.')
    parser.add_argument('--shuffle', action='store_true', default=False,
                        help='Whether to shuffle the data.')
    args = parser.parse_args()

    data = GazeCapture(root=os.path.join(args.root, args.split), size=(128, 128), verbose=True)
    loader = create_dataloader(
        data,
        batch_size=args.batch_size,
        shuffle=args.shuffle, seed=None
    )
