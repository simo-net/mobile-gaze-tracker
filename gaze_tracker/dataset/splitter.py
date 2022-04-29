import os
import json
import shutil
import numpy as np
from glob import glob
from multiprocessing import Process
from sklearn.model_selection import train_test_split


# ----------------------------------------------------------------------------------------------------------------------

def google_splitter(files: list, out_root: str):
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

    for i in files:
        expt_name = i.split('/')[-2]  # 00002, 00011, ...

        with open(os.path.join(i, 'appleFace.json')) as f:
            face_det = json.load(f)  # keys: 'H', 'W', 'X', 'Y', 'IsValid'
        with open(os.path.join(i, 'appleLeftEye.json')) as f:
            l_eye_det = json.load(f)  # keys: 'H', 'W', 'X', 'Y', 'IsValid'
        with open(os.path.join(i, 'appleRightEye.json')) as f:
            r_eye_det = json.load(f)  # keys: 'H', 'W', 'X', 'Y', 'IsValid'
        with open(os.path.join(i, 'dotInfo.json')) as f:
            dot = json.load(f)  # keys: 'DotNum', 'XPts', 'YPts', 'XCam', 'YCam', 'Time'

        # with open(os.path.join(i, 'faceGrid.json')) as f:
        #     facegrid = json.load(f)  # keys: 'X', 'Y', 'W', 'H', 'IsValid'
        # with open(os.path.join(i, 'frames.json')) as f:
        #     frames = json.load(f)  # E.G.: 00000.jpg, 00001.jpg, ...
        # with open(os.path.join(i, 'motion.json')) as f:
        #     motion = json.load(f)  # list of dicts with keys 'GravityX', 'UserAcceleration', 'AttitudeRotationMatrix', 'AttitudePitch', 'Time', 'AttitudeQuaternion', 'AttitudeRoll', 'RotationRate', 'AttitudeYaw', 'DotNum'

        with open(os.path.join(i, 'screen.json')) as f:
            screen_info = json.load(f)  # keys: 'H', 'W', 'Orientation'
        with open(os.path.join(i, 'info.json')) as f:
            info = json.load(f)  # keys: 'Dataset', 'DeviceName', 'TotalFrames', 'NumFaceDetections', 'NumEyeDetections'
            device = info['DeviceName'].replace(' ', '')  # E.G.: iPhone5, iPhone6, ...
        if not ('iPhone' in device):
            continue

        portrait_orientation = (np.asarray(screen_info["Orientation"]) == 1).astype(bool)  # it can also be 2, 3, 4, ...
        l_eye_valid = np.array(l_eye_det['IsValid'], dtype=bool)
        r_eye_valid = np.array(r_eye_det['IsValid'], dtype=bool)
        valid_mask = l_eye_valid & r_eye_valid & portrait_orientation

        all_dots = np.asarray(dot['DotNum'])
        valid_dots = np.unique(all_dots[np.where(valid_mask)])

        try:
            train_dots, test_dots = train_test_split(valid_dots, test_size=0.2)
        except:
            continue  # Too few training dots to split in train and test, skip participant
        try:
            test_dots, val_dots = train_test_split(test_dots, test_size=0.4)
        except:
            val_dots = []  # Too few test dots to split in test and val, no validation dots for such participant

        split_mask = np.zeros(len(valid_mask), dtype=np.uint8)
        split_choice = {1: 'train', 2: 'test', 3: 'val'}
        for j in train_dots:
            split_mask[all_dots == j] = 1
        for j in test_dots:
            split_mask[all_dots == j] = 2
        for j in val_dots:
            split_mask[all_dots == j] = 3
        split_mask = split_mask * valid_mask

        valid_ids = np.where(split_mask > 0)[0]

        for frame_idx in valid_ids:
            fname = str(frame_idx).zfill(5)

            split = split_choice.get(split_mask[frame_idx])
            out_dir = os.path.join(out_root, split)
            os.makedirs(os.path.join(out_dir, 'images'), exist_ok=True)
            os.makedirs(os.path.join(out_dir, 'meta'), exist_ok=True)

            meta = {
                'device': device,
                'screen_h': screen_info["H"][frame_idx],
                'screen_w': screen_info["W"][frame_idx],
                'face_valid': face_det["IsValid"][frame_idx],
                'face_x': round(face_det['X'][frame_idx]),
                'face_y': round(face_det['Y'][frame_idx]),
                'face_w': round(face_det['W'][frame_idx]),
                'face_h': round(face_det['H'][frame_idx]),
                'leye_x': round(l_eye_det['X'][frame_idx]),
                'leye_y': round(l_eye_det['Y'][frame_idx]),
                'leye_w': round(l_eye_det['W'][frame_idx]),
                'leye_h': round(l_eye_det['H'][frame_idx]),
                'reye_x': round(r_eye_det['X'][frame_idx]),
                'reye_y': round(r_eye_det['Y'][frame_idx]),
                'reye_w': round(r_eye_det['W'][frame_idx]),
                'reye_h': round(r_eye_det['H'][frame_idx]),
                'dot_x_cam': dot['XCam'][frame_idx],
                'dot_y_cam': dot['YCam'][frame_idx],
                'dot_x_pix': dot['XPts'][frame_idx],
                'dot_y_pix': dot['YPts'][frame_idx],
            }
            meta['leye_x'] += meta['face_x']
            meta['leye_y'] += meta['face_y']
            meta['reye_x'] += meta['face_x']
            meta['reye_y'] += meta['face_y']

            meta_file = os.path.join(out_dir, 'meta', expt_name + '__' + fname + '.json')
            with open(meta_file, 'w') as outfile:
                json.dump(meta, outfile)
            shutil.copy(os.path.join(i, 'frames', fname + '.jpg'),
                        os.path.join(out_dir, 'images', expt_name + '__' + fname + '.jpg'))

        print(f"{i.split('/')[-1]} folder completed --> {len(valid_ids)}/{len(valid_mask)} valid images")

    return 0


# ----------------------------------------------------------------------------------------------------------------------

def mit_splitter(files: list, out_root: str):
    """
    Convert the Gaze Capture dataset (and prepare it for easy use in PyTorch) by following the pre-processing procedure
    that was used in the original paper announcing dataset release:
    "Eye Tracking for Everyone", CVPR 2016.

    Steps:
    1. Keep only portrait orientation images
    2. Create metadata file for each image
    3. Split data based on MIT annotation (separate participants in train, test, val)
    """

    for i in files:
        expt_name = i.split('/')[-2]  # 00002, 00011, ...

        with open(os.path.join(i, 'appleFace.json')) as f:
            face_det = json.load(f)  # keys: 'H', 'W', 'X', 'Y', 'IsValid'
        with open(os.path.join(i, 'appleLeftEye.json')) as f:
            l_eye_det = json.load(f)  # keys: 'H', 'W', 'X', 'Y', 'IsValid'
        with open(os.path.join(i, 'appleRightEye.json')) as f:
            r_eye_det = json.load(f)  # keys: 'H', 'W', 'X', 'Y', 'IsValid'
        with open(os.path.join(i, 'dotInfo.json')) as f:
            dot = json.load(f)  # keys: 'DotNum', 'XPts', 'YPts', 'XCam', 'YCam', 'Time'

        # with open(os.path.join(i, 'faceGrid.json')) as f:
        #     facegrid = json.load(f)  # keys: 'X', 'Y', 'W', 'H', 'IsValid'
        # with open(os.path.join(i, 'frames.json')) as f:
        #     frames = json.load(f)  # E.G.: 00000.jpg, 00001.jpg, ...
        # with open(os.path.join(i, 'motion.json')) as f:
        #     motion = json.load(f)  # list of dicts with keys 'GravityX', 'UserAcceleration', 'AttitudeRotationMatrix', 'AttitudePitch', 'Time', 'AttitudeQuaternion', 'AttitudeRoll', 'RotationRate', 'AttitudeYaw', 'DotNum'

        with open(os.path.join(i, 'screen.json')) as f:
            screen_info = json.load(f)  # keys: 'H', 'W', 'Orientation'
        with open(os.path.join(i, 'info.json')) as f:
            info = json.load(f)  # keys: 'Dataset', 'DeviceName', 'TotalFrames', 'NumFaceDetections', 'NumEyeDetections'
            split = info['Dataset']  # E.G.: train, test or val
            device = info['DeviceName'].replace(' ', '')  # E.G.: iPhone5, iPhone6, ...

        portrait_orientation = (np.asarray(screen_info["Orientation"]) == 1).astype(bool)  # it can also be 2, 3, 4, ...
        l_eye_valid = np.array(l_eye_det['IsValid'], dtype=bool)
        r_eye_valid = np.array(r_eye_det['IsValid'], dtype=bool)
        valid_mask = l_eye_valid & r_eye_valid & portrait_orientation

        valid_ids = np.where(valid_mask)[0]

        out_dir = os.path.join(out_root, split)
        os.makedirs(os.path.join(out_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'meta'), exist_ok=True)
        for frame_idx in valid_ids:
            fname = str(frame_idx).zfill(5)

            meta = {
                'device': device,
                'screen_h': screen_info["H"][frame_idx],
                'screen_w': screen_info["W"][frame_idx],
                'face_valid': face_det["IsValid"][frame_idx],
                'face_x': round(face_det['X'][frame_idx]),
                'face_y': round(face_det['Y'][frame_idx]),
                'face_w': round(face_det['W'][frame_idx]),
                'face_h': round(face_det['H'][frame_idx]),
                'leye_x': round(l_eye_det['X'][frame_idx]),
                'leye_y': round(l_eye_det['Y'][frame_idx]),
                'leye_w': round(l_eye_det['W'][frame_idx]),
                'leye_h': round(l_eye_det['H'][frame_idx]),
                'reye_x': round(r_eye_det['X'][frame_idx]),
                'reye_y': round(r_eye_det['Y'][frame_idx]),
                'reye_w': round(r_eye_det['W'][frame_idx]),
                'reye_h': round(r_eye_det['H'][frame_idx]),
                'dot_x_cam': dot['XCam'][frame_idx],
                'dot_y_cam': dot['YCam'][frame_idx],
                'dot_x_pix': dot['XPts'][frame_idx],
                'dot_y_pix': dot['YPts'][frame_idx],
            }
            meta['leye_x'] += meta['face_x']
            meta['leye_y'] += meta['face_y']
            meta['reye_x'] += meta['face_x']
            meta['reye_y'] += meta['face_y']

            meta_file = os.path.join(out_dir, 'meta', expt_name + '__' + fname + '.json')
            with open(meta_file, 'w') as outfile:
                json.dump(meta, outfile)
            shutil.copy(os.path.join(i, 'frames', fname + '.jpg'),
                        os.path.join(out_dir, 'images', expt_name + '__' + fname + '.jpg'))

        print(f"{i.split('/')[-1]} folder completed --> {len(valid_ids)}/{len(valid_mask)} valid images")

    return 0


# **********************************************************************************************************************

def split_data(in_dir: str, out_dir: str, splitter_function: object,
               workers: int = 2):
    # Example of usage:
    # split_data(in_dir='/path/to/data', out_dir='/path/for/preprocessed/data', splitter=mit_splitter, workers=6)
    # split_data(in_dir='/path/to/data', out_dir='/path/for/preprocessed/data', splitter=google_splitter, workers=6)
    procs = []
    files = glob(os.path.join(in_dir, "*"))
    chunk = len(files) // workers
    for w in range(workers):
        files_chunk = files[w*chunk: (w+1)*chunk]
        if w == workers - 1:
            files_chunk = files[w*chunk:]

        proc = Process(target=splitter_function, args=(files_chunk, out_dir))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()
