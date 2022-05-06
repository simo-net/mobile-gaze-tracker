import os
import cv2
import json
import dlib
from glob import glob
from multiprocessing import Process
import numpy as np


FACIAL_LANDMARKS_DLIB = dict([
	("mouth", (48, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 35)),
	("jaw", (0, 17))
])


def in_box(box, point):
    x1, y1, x2, y2 = box
    x, y = point
    if x1 < x < x2 and y1 < y < y2:
        return True
    return False


def kps_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((shape.num_parts, 2), dtype=dtype)

	# loop over all facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for k in range(shape.num_parts):
		coords[k] = (shape.part(k).x, shape.part(k).y)

	# return the list of (x, y)-coordinates
	return coords


def keypoints_adder(files: list, p: str):
    """
    Add eye landmarks to the metafiles of the Gaze Capture Dataset using DLib.
    """

    # Initialize dlib's face detector (based on HOGs + linear SVM) and then create the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)

    margin = 10
    no_face, err_ctr = 0, 0
    for i in files:
        img = cv2.imread(i)
        bw_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        meta_file = i.replace('images', 'meta').replace('.jpg', '.json')
        with open(meta_file, 'r') as f:
            meta = json.load(f)

        # Detect face if no valid face info is present in the metadata
        leye_x, leye_y, leye_w, leye_h = meta['leye_x'], meta['leye_y'], meta['leye_w'], meta['leye_h']
        reye_x, reye_y, reye_w, reye_h = meta['reye_x'], meta['reye_y'], meta['reye_w'], meta['reye_h']
        if meta['face_valid']:
            fx, fy, fw, fh = meta['face_x'], meta['face_y'], meta['face_w'], meta['face_h']
        else:
            no_face += 1
            faces = detector(bw_img, 1)
            if not faces:
                continue
            eye_up_bound, eye_low_bound = min([leye_y, reye_y]), max([leye_y + leye_h, reye_y + reye_h])
            for face in faces:
                # check that the detected face comprehends all eye points
                if face.left() < reye_x and face.right() > leye_x+leye_w and face.top() < eye_up_bound and face.bottom() > eye_low_bound:
                    fx, fy, fw, fh = face.left(), face.top(), face.right() - face.left(), face.bottom() - face.top()
                    break

        # Refine the eye ROIs
        kps = kps_to_np(predictor(bw_img, dlib.rectangle(fx, fy, fx+fw, fy+fh)))
        kps_reye = kps[slice(*FACIAL_LANDMARKS_DLIB['right_eye'], 1)]
        reye_x_new, reye_y_new, reye_w_new, reye_h_new = cv2.boundingRect(kps_reye)
        reye_x_new, reye_y_new = reye_x_new - margin, reye_y_new - margin
        reye_w_new, reye_h_new = reye_w_new + 2*margin, reye_h_new + 2*margin
        kps_leye = kps[slice(*FACIAL_LANDMARKS_DLIB['left_eye'], 1)]
        leye_x_new, leye_y_new, leye_w_new, leye_h_new = cv2.boundingRect(kps_leye)
        leye_x_new, leye_y_new = leye_x_new - margin, leye_y_new - margin
        leye_w_new, leye_h_new = leye_w_new + 2*margin, leye_h_new + 2*margin

        # Make sure the detected landmark points are within the original eye bounding boxes (from metadata)
        eye_box = (reye_x - margin//2, reye_y - margin//2,
                   reye_x + reye_w + margin//2, reye_y + reye_h + margin//2)
        if in_box(eye_box, (reye_x_new, reye_y_new) and in_box(eye_box, (reye_x_new+reye_w_new, reye_y_new+reye_h_new))):
            meta['reye_x1'], meta['reye_y1'] = reye_x_new, reye_y_new
            meta['reye_x2'], meta['reye_y2'] = reye_x_new + reye_w_new, reye_y_new + reye_h_new
        else:
            err_ctr += 1
            meta['reye_x1'], meta['reye_y1'] = reye_x, reye_y
            meta['reye_x2'], meta['reye_y2'] = reye_x + reye_w, reye_y + reye_h
        if in_box(eye_box, (leye_x_new, leye_y_new) and in_box(eye_box, (leye_x_new+leye_w_new, leye_y_new+leye_h_new))):
            meta['leye_x1'], meta['leye_y1'] = leye_x_new, leye_y_new
            meta['leye_x2'], meta['leye_y2'] = leye_x_new + leye_w_new, leye_y_new + leye_h_new
        else:
            err_ctr += 1
            meta['leye_x1'], meta['leye_y1'] = leye_x, leye_y
            meta['leye_x2'], meta['leye_y2'] = leye_x + leye_w, leye_y + leye_h

        with open(meta_file, 'w') as outfile:
            json.dump(meta, outfile)

    print(f"There were {no_face} invalid faces in metadata and {err_ctr} out-of-box detected eye-landmarks.")


# **********************************************************************************************************************

def add_keypoints(in_dir: str, p: str,
                  workers: int = 2):
    procs = []
    files = glob(os.path.join(in_dir, "*", "images", "*.jpg"))
    chunk = len(files) // workers
    for w in range(workers):
        files_chunk = files[w*chunk: (w+1)*chunk]
        if w == workers - 1:
            files_chunk = files[w*chunk:]

        proc = Process(target=keypoints_adder, args=(files_chunk, p))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()
