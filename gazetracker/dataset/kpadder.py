import os
import cv2
import json
import dlib
import numpy as np
from glob import glob
from multiprocessing import Process


FACIAL_LANDMARKS_DLIB = dict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 35)),
    ("jaw", (0, 17))
])


def kps_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)
    # loop over all facial landmarks and convert them to a 2-tuple of (x, y)-coordinates
    for k in range(shape.num_parts):
        coords[k] = (shape.part(k).x, shape.part(k).y)
    # return the list of (x, y)-coordinates
    return coords


def in_box(box, point):
    (x1, y1), (x2, y2) = box
    x, y = point
    if x1 < x < x2 and y1 < y < y2:
        return True
    return False


def find_center(box):
    (x1, y1), (x2, y2) = box
    w, h = x2-x1, y2-y1
    return x1 + w//2, y1 + h//2


def box_from_center(center, box_shape, img_shape):
    x_center, y_center = center
    w_box, h_box = box_shape
    w_img, h_img = img_shape
    x1, y1 = x_center - w_box//2, y_center - h_box//2
    x2, y2 = x1 + w_box, y1 + h_box
    if x1 < 0: (x1, x2) = (0, x1+w_box)
    if x2 > w_img-1: (x2, x1) = (w_img-1, w_img-1-w_box)
    if y1 < 0: (y1, y2) = (0, y1+h_box)
    if y2 > h_img-1: (y2, y1) = (h_img-1, h_img-1-h_box)
    return [(x1, y1), (x2, y2)]


def box_with_margin(box, margins):
    (x1, y1), (x2, y2) = box
    w, h = x2 - x1, y2 - y1
    x_margin, y_margin = margins
    w_new, h_new = w + 2 * x_margin, h + 2 * y_margin
    x1_new, y1_new = x1 - x_margin, y1 - y_margin
    x2_new, y2_new = x1_new + w_new, y1_new + h_new
    return [(x1_new, y1_new), (x2_new, y2_new)]


def keypoints_adder(files: list, p: str,
                    target_shape: (int, int)):
    """
    Add eye landmarks to the metafiles of the Gaze Capture Dataset using DLib.
    """

    # Initialize dlib's face detector (based on HOGs + linear SVM) and then create the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)

    # Initialize variables
    no_face, eye_errors = 0, []
    W, H = target_shape

    # Loop through all files
    for img_file in files:
        split = img_file.split('/')[-3]
        file_name = img_file.split('/')[-1][:-4]

        # Grub image and metadata
        bw_img = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2GRAY)
        h_img, w_img = bw_img.shape
        meta_file = img_file.replace('images', 'meta').replace('.jpg', '.json')
        with open(meta_file, 'r') as f:
            meta = json.load(f)

        # Take the right and left reference eye-landmarks from metadata
        reye_x1_ref, reye_y1_ref, reye_w_ref, reye_h_ref = meta['reye_x'], meta['reye_y'], meta['reye_w'], meta['reye_h']
        reye_x2_ref, reye_y2_ref = reye_x1_ref + reye_w_ref, reye_y1_ref + reye_h_ref
        leye_x1_ref, leye_y1_ref, leye_w_ref, leye_h_ref = meta['leye_x'], meta['leye_y'], meta['leye_w'], meta['leye_h']
        leye_x2_ref, leye_y2_ref = leye_x1_ref + leye_w_ref, leye_y1_ref + leye_h_ref

        # Detect face if no valid face info is present in the metadata
        if meta['face_valid']:
            fx, fy, fw, fh = meta['face_x'], meta['face_y'], meta['face_w'], meta['face_h']
        else:
            no_face += 1
            print('No valid face detected in image for file ' + file_name + ' in ' + split + ' set')
            faces = detector(bw_img, 1)
            if not faces:
                continue
            eye_up_bound, eye_low_bound = min([leye_y1_ref, reye_y1_ref]), max([leye_y2_ref, reye_y2_ref])
            for face in faces:
                # check that the detected face comprehends all eye points
                if face.left() < reye_x1_ref and face.right() > leye_x2_ref and face.top() < eye_up_bound and face.bottom() > eye_low_bound:
                    fx, fy, fw, fh = face.left(), face.top(), face.right() - face.left(), face.bottom() - face.top()
                    break

        # Refine eye ROIs from face info
        kps = kps_to_np(predictor(bw_img, dlib.rectangle(fx, fy, fx + fw, fy + fh)))
        kps_reye = kps[slice(*FACIAL_LANDMARKS_DLIB['right_eye'], 1)]
        reye_x1, reye_y1, reye_w, reye_h = cv2.boundingRect(kps_reye)
        reye_x2, reye_y2 = reye_x1 + reye_w, reye_y1 + reye_h
        kps_leye = kps[slice(*FACIAL_LANDMARKS_DLIB['left_eye'], 1)]
        leye_x1, leye_y1, leye_w, leye_h = cv2.boundingRect(kps_leye)
        leye_x2, leye_y2 = leye_x1 + leye_w, leye_y1 + leye_h

        # Resize the new eye ROIs to the target shape
        (reye_x1, reye_y1), (reye_x2, reye_y2) = box_from_center(
            center=find_center(box=[(reye_x1, reye_y1), (reye_x2, reye_y2)]),
            box_shape=(W, H), img_shape=(w_img, h_img))
        (leye_x1, leye_y1), (leye_x2, leye_y2) = box_from_center(
            center=find_center(box=[(leye_x1, leye_y1), (leye_x2, leye_y2)]),
            box_shape=(W, H), img_shape=(w_img, h_img))

        # Make sure the detected landmark points are within the image and the original eye bounding boxes (from metadata)
        img_box = [(0, 0), (w_img, h_img)]
        if in_box(img_box, (reye_x1, reye_y1)) and in_box(img_box, (reye_x2, reye_y2)):
            meta['reye_x1'], meta['reye_y1'] = reye_x1, reye_y1
            meta['reye_x2'], meta['reye_y2'] = reye_x2, reye_y2
        else:
            (reye_x1_ref, reye_y1_ref), (reye_x2_ref, reye_y2_ref) = box_from_center(
                center=find_center(box=[(reye_x1_ref, reye_y1_ref), (reye_x2_ref, reye_y2_ref)]),
                box_shape=(W, H), img_shape=(w_img, h_img))  # Resize the right reference eye ROI to the target shape
            meta['reye_x1'], meta['reye_y1'] = reye_x1_ref, reye_y1_ref
            meta['reye_x2'], meta['reye_y2'] = reye_x2_ref, reye_y2_ref
            print('Right eye landmark is not in image boundary for file ' + file_name + ' in ' + split + ' set')
            eye_errors.append(img_file)
        if in_box(img_box, (leye_x1, leye_y1)) and in_box(img_box, (leye_x2, leye_y2)):
            meta['leye_x1'], meta['leye_y1'] = leye_x1, leye_y1
            meta['leye_x2'], meta['leye_y2'] = leye_x2, leye_y2
        else:
            (leye_x1_ref, leye_y1_ref), (leye_x2_ref, leye_y2_ref) = box_from_center(
                center=find_center(box=[(leye_x1_ref, leye_y1_ref), (leye_x2_ref, leye_y2_ref)]),
                box_shape=(W, H), img_shape=(w_img, h_img))  # Resize the left reference eye ROI to the target shape
            meta['leye_x1'], meta['leye_y1'] = leye_x1_ref, leye_y1_ref
            meta['leye_x2'], meta['leye_y2'] = leye_x2_ref, leye_y2_ref
            print('Left eye landmark is not in image boundary for file ' + file_name + ' in ' + split + ' set')
            eye_errors.append(img_file)

        # Update metadata according to the new eye landmarks
        meta_file = meta_file[:-5] + '_new.json'  # TODO: remove this and overwrite the meta file!!!
        with open(meta_file, 'w') as outfile:
            json.dump(meta, outfile)

    eye_errors = sorted(list(set(eye_errors)))
    print(f"There were {no_face} invalid faces in metadata and {len(eye_errors)} out-of-image or out-of-eye detected eye-landmarks.")
    return eye_errors


def keypoints_adder_check_reference(files: list, p: str,
                                    target_shape: (int, int)):
    """
    Add eye landmarks to the metafiles of the Gaze Capture Dataset using DLib.
    """

    # Initialize dlib's face detector (based on HOGs + linear SVM) and then create the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)

    # Initialize variables
    margin = 30
    no_face, eye_errors = 0, []
    W, H = target_shape

    # Loop through all files
    for img_file in files:
        split = img_file.split('/')[-3]
        file_name = img_file.split('/')[-1][:-4]

        # Grub image and metadata
        bw_img = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2GRAY)
        h_img, w_img = bw_img.shape
        meta_file = img_file.replace('images', 'meta').replace('.jpg', '.json')
        with open(meta_file, 'r') as f:
            meta = json.load(f)

        # Take the right and left reference eye-landmarks from metadata
        reye_x1_ref, reye_y1_ref, reye_w_ref, reye_h_ref = meta['reye_x'], meta['reye_y'], meta['reye_w'], meta['reye_h']
        reye_x2_ref, reye_y2_ref = reye_x1_ref + reye_w_ref, reye_y1_ref + reye_h_ref
        leye_x1_ref, leye_y1_ref, leye_w_ref, leye_h_ref = meta['leye_x'], meta['leye_y'], meta['leye_w'], meta['leye_h']
        leye_x2_ref, leye_y2_ref = leye_x1_ref + leye_w_ref, leye_y1_ref + leye_h_ref

        # Detect face if no valid face info is present in the metadata
        if meta['face_valid']:
            fx, fy, fw, fh = meta['face_x'], meta['face_y'], meta['face_w'], meta['face_h']
        else:
            no_face += 1
            print('No valid face detected in image for file ' + file_name + ' in ' + split + ' set')
            faces = detector(bw_img, 1)
            if not faces:
                continue
            eye_up_bound, eye_low_bound = min([leye_y1_ref, reye_y1_ref]), max([leye_y2_ref, reye_y2_ref])
            for face in faces:
                # check that the detected face comprehends all eye points
                if face.left() < reye_x1_ref and face.right() > leye_x2_ref and face.top() < eye_up_bound and face.bottom() > eye_low_bound:
                    fx, fy, fw, fh = face.left(), face.top(), face.right() - face.left(), face.bottom() - face.top()
                    break

        # Refine eye ROIs from face info
        kps = kps_to_np(predictor(bw_img, dlib.rectangle(fx, fy, fx + fw, fy + fh)))
        kps_reye = kps[slice(*FACIAL_LANDMARKS_DLIB['right_eye'], 1)]
        reye_x1, reye_y1, reye_w, reye_h = cv2.boundingRect(kps_reye)
        reye_x2, reye_y2 = reye_x1 + reye_w, reye_y1 + reye_h
        kps_leye = kps[slice(*FACIAL_LANDMARKS_DLIB['left_eye'], 1)]
        leye_x1, leye_y1, leye_w, leye_h = cv2.boundingRect(kps_leye)
        leye_x2, leye_y2 = leye_x1 + leye_w, leye_y1 + leye_h

        # Resize the new eye ROIs to the target shape
        (reye_x1, reye_y1), (reye_x2, reye_y2) = box_from_center(
            center=find_center(box=[(reye_x1, reye_y1), (reye_x2, reye_y2)]),
            box_shape=(W, H), img_shape=(w_img, h_img))
        (leye_x1, leye_y1), (leye_x2, leye_y2) = box_from_center(
            center=find_center(box=[(leye_x1, leye_y1), (leye_x2, leye_y2)]),
            box_shape=(W, H), img_shape=(w_img, h_img))

        # Resize the reference eye ROIs to the target shape
        (reye_x1_ref, reye_y1_ref), (reye_x2_ref, reye_y2_ref) = box_from_center(
            center=find_center(box=[(reye_x1_ref, reye_y1_ref), (reye_x2_ref, reye_y2_ref)]),
            box_shape=(W, H), img_shape=(w_img, h_img))
        (leye_x1_ref, leye_y1_ref), (leye_x2_ref, leye_y2_ref) = box_from_center(
            center=find_center(box=[(leye_x1_ref, leye_y1_ref), (leye_x2_ref, leye_y2_ref)]),
            box_shape=(W, H), img_shape=(w_img, h_img))

        # Make sure the detected landmark points are within the image and the original eye bounding boxes (from metadata)
        img_box = [(0, 0), (w_img, h_img)]
        meta['reye_x1'], meta['reye_y1'] = reye_x1_ref, reye_y1_ref
        meta['reye_x2'], meta['reye_y2'] = reye_x2_ref, reye_y2_ref
        if in_box(img_box, (reye_x1, reye_y1)) and in_box(img_box, (reye_x2, reye_y2)):
            reye_box = [(reye_x1_ref - margin, reye_y1_ref - margin), (reye_x2_ref + margin, reye_y2_ref + margin)]
            if in_box(reye_box, (reye_x1, reye_y1)) and in_box(reye_box, (reye_x2, reye_y2)):
                meta['reye_x1'], meta['reye_y1'] = reye_x1, reye_y1
                meta['reye_x2'], meta['reye_y2'] = reye_x2, reye_y2
            else:
                print('Right eye landmark is not in reference boundary for file ' + file_name + ' in ' + split + ' set')
                eye_errors.append(img_file)
        else:
            print('Right eye landmark is not in image boundary for file ' + file_name + ' in ' + split + ' set')
            eye_errors.append(img_file)
        meta['leye_x1'], meta['leye_y1'] = leye_x1_ref, leye_y1_ref
        meta['leye_x2'], meta['leye_y2'] = leye_x2_ref, leye_y2_ref
        if in_box(img_box, (leye_x1, leye_y1)) and in_box(img_box, (leye_x2, leye_y2)):
            leye_box = [(leye_x1_ref - margin, leye_y1_ref - margin), (leye_x2_ref + margin, leye_y2_ref + margin)]
            if in_box(leye_box, (leye_x1, leye_y1)) and in_box(leye_box, (leye_x2, leye_y2)):
                meta['leye_x1'], meta['leye_y1'] = leye_x1, leye_y1
                meta['leye_x2'], meta['leye_y2'] = leye_x2, leye_y2
            else:
                print('Left eye landmark is not in reference boundary for file ' + file_name + ' in ' + split + ' set')
                eye_errors.append(img_file)
        else:
            print('Left eye landmark is not in image boundary for file ' + file_name + ' in ' + split + ' set')
            eye_errors.append(img_file)

        # Update metadata according to the new eye landmarks
        meta_file = meta_file[:-5] + '_new.json'  # TODO: remove this and overwrite the meta file!!!
        with open(meta_file, 'w') as outfile:
            json.dump(meta, outfile)

    eye_errors = sorted(list(set(eye_errors)))
    print(f"There were {no_face} invalid faces in metadata and {len(eye_errors)} out-of-image or out-of-eye detected eye-landmarks.")
    return eye_errors


def keypoints_adder_different_shape(files: list, p: str):
    """
    Add eye landmarks to the metafiles of the Gaze Capture Dataset using DLib.
    """

    # Initialize dlib's face detector (based on HOGs + linear SVM) and then create the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)

    # Initialize variables
    margin = 10
    no_face, err_ctr = 0, 0

    # Loop through files
    for img_file in files:

        # Grub image and metadata
        bw_img = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2GRAY)
        meta_file = img_file.replace('images', 'meta').replace('.jpg', '.json')
        with open(meta_file, 'r') as f:
            meta = json.load(f)

        # Take the right and left reference eye-landmarks from metadata
        reye_x1_ref, reye_y1_ref, reye_w_ref, reye_h_ref = meta['reye_x'], meta['reye_y'], meta['reye_w'], meta['reye_h']
        reye_x2_ref, reye_y2_ref = reye_x1_ref + reye_w_ref, reye_y1_ref + reye_h_ref
        leye_x1_ref, leye_y1_ref, leye_w_ref, leye_h_ref = meta['leye_x'], meta['leye_y'], meta['leye_w'], meta['leye_h']
        leye_x2_ref, leye_y2_ref = leye_x1_ref + leye_w_ref, leye_y1_ref + leye_h_ref

        # Detect face if no valid face info is present in the metadata
        if meta['face_valid']:
            fx, fy, fw, fh = meta['face_x'], meta['face_y'], meta['face_w'], meta['face_h']
        else:
            no_face += 1
            print('No valid face!')
            faces = detector(bw_img, 1)
            if not faces:
                continue
            eye_up_bound, eye_low_bound = min([leye_y1_ref, reye_y1_ref]), max([leye_y2_ref, reye_y2_ref])
            for face in faces:
                # check that the detected face comprehends all eye points
                if face.left() < reye_x1_ref and face.right() > leye_x2_ref and face.top() < eye_up_bound and face.bottom() > eye_low_bound:
                    fx, fy, fw, fh = face.left(), face.top(), face.right() - face.left(), face.bottom() - face.top()
                    break

        # Refine eye ROIs from face info
        kps = kps_to_np(predictor(bw_img, dlib.rectangle(fx, fy, fx + fw, fy + fh)))
        kps_reye = kps[slice(*FACIAL_LANDMARKS_DLIB['right_eye'], 1)]
        reye_x1, reye_y1, reye_w, reye_h = cv2.boundingRect(kps_reye)
        reye_x2, reye_y2 = reye_x1 + reye_w, reye_y1 + reye_h
        kps_leye = kps[slice(*FACIAL_LANDMARKS_DLIB['left_eye'], 1)]
        leye_x1, leye_y1, leye_w, leye_h = cv2.boundingRect(kps_leye)
        leye_x2, leye_y2 = leye_x1 + leye_w, leye_y1 + leye_h

        # Add a margin to the new eye ROIs
        (reye_x1, reye_y1), (reye_x2, reye_y2) = box_with_margin(box=[(reye_x1, reye_y1), (reye_x2, reye_y2)],
                                                                 margins=(margin, margin))
        (leye_x1, leye_y1), (leye_x2, leye_y2) = box_with_margin(box=[(leye_x1, leye_y1), (leye_x2, leye_y2)],
                                                                 margins=(margin, margin))

        # Make sure the detected landmark points are within the original eye bounding boxes (from metadata)
        reye_box = [(reye_x1_ref - margin, reye_y1_ref - margin), (reye_x2_ref + margin, reye_y2_ref + margin)]
        leye_box = [(leye_x1_ref - margin, leye_y1_ref - margin), (leye_x2_ref + margin, leye_y2_ref + margin)]
        if in_box(reye_box, (reye_x1, reye_y1)) and in_box(reye_box, (reye_x2, reye_y2)):
            meta['reye_x1'], meta['reye_y1'] = reye_x1, reye_y1
            meta['reye_x2'], meta['reye_y2'] = reye_x2, reye_y2
        else:
            err_ctr += 1
            meta['reye_x1'], meta['reye_y1'] = reye_x1_ref, reye_y1_ref
            meta['reye_x2'], meta['reye_y2'] = reye_x2_ref, reye_y2_ref
        if in_box(leye_box, (leye_x1, leye_y1)) and in_box(leye_box, (leye_x2, leye_y2)):
            meta['leye_x1'], meta['leye_y1'] = leye_x1, leye_y1
            meta['leye_x2'], meta['leye_y2'] = leye_x2, leye_y2
        else:
            err_ctr += 1
            meta['leye_x1'], meta['leye_y1'] = leye_x1_ref, leye_y1_ref
            meta['leye_x2'], meta['leye_y2'] = leye_x2_ref, leye_y2_ref

        # Update metadata according to the new eye landmarks
        meta_file = meta_file[:-5] + '_new.json'  # TODO: remove this and overwrite the meta file!!!
        with open(meta_file, 'w') as outfile:
            json.dump(meta, outfile)

    print(f"There were {no_face} invalid faces in metadata and {err_ctr} out-of-box detected eye-landmarks.")


# **********************************************************************************************************************

def add_keypoints(in_dir: str, p: str,
                  workers: int = 2):
    target_shape = (128, 128)  # (W, H)
    procs = []
    files = sorted(glob(os.path.join(in_dir, "*", "images", "*.jpg")))
    chunk = len(files) // workers
    for w in range(workers):
        files_chunk = files[w * chunk: (w + 1) * chunk]
        if w == workers - 1:
            files_chunk = files[w * chunk:]

        proc = Process(target=keypoints_adder, args=(files_chunk, p, target_shape))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()


def add_keypoints_different_shape(in_dir: str, p: str,
                                  workers: int = 2):
    procs = []
    files = sorted(glob(os.path.join(in_dir, "*", "images", "*.jpg")))
    chunk = len(files) // workers
    for w in range(workers):
        files_chunk = files[w * chunk: (w + 1) * chunk]
        if w == workers - 1:
            files_chunk = files[w * chunk:]

        proc = Process(target=keypoints_adder_different_shape, args=(files_chunk, p))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()

