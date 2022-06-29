import os
import cv2
import json
import dlib
import random
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches

num_samples = 20

in_dir = '/home/simonetunige/data/mobile-gaze/gaze-capture_google-split'
files = glob(os.path.join(in_dir, "*", "images", "*.jpg"))
random.shuffle(files)
files = files[:num_samples]

p = "./checkpoints/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)


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


margin = 10
no_face, err_ctr = 0, 0
for i in files:
    print('*'*30)

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
        print('No valid face!')
        faces = detector(bw_img, 1)
        if not faces:
            continue
        eye_up_bound, eye_low_bound = min([leye_y, reye_y]), max([leye_y + leye_h, reye_y + reye_h])
        for face in faces:
            # check that the detected face comprehends all eye points
            if face.left() < reye_x and face.right() > leye_x + leye_w and face.top() < eye_up_bound and face.bottom() > eye_low_bound:
                fx, fy, fw, fh = face.left(), face.top(), face.right() - face.left(), face.bottom() - face.top()
                break

    # Refine the eye ROIs
    kps = kps_to_np(predictor(bw_img, dlib.rectangle(fx, fy, fx + fw, fy + fh)))
    kps_reye = kps[slice(*FACIAL_LANDMARKS_DLIB['right_eye'], 1)]
    reye_x_new, reye_y_new, reye_w_new, reye_h_new = cv2.boundingRect(kps_reye)
    reye_x_new, reye_y_new = reye_x_new - margin, reye_y_new - margin
    reye_w_new, reye_h_new = reye_w_new + 2 * margin, reye_h_new + 2 * margin
    kps_leye = kps[slice(*FACIAL_LANDMARKS_DLIB['left_eye'], 1)]
    leye_x_new, leye_y_new, leye_w_new, leye_h_new = cv2.boundingRect(kps_leye)
    leye_x_new, leye_y_new = leye_x_new - margin, leye_y_new - margin
    leye_w_new, leye_h_new = leye_w_new + 2 * margin, leye_h_new + 2 * margin

    # Make sure the detected landmark points are within the original eye bounding boxes (from metadata)
    reye_box = (reye_x - margin, reye_y - margin,
                reye_x + reye_w + margin, reye_y + reye_h + margin)
    if in_box(reye_box, (reye_x_new, reye_y_new)) and in_box(reye_box, (reye_x_new + reye_w_new, reye_y_new + reye_h_new)):
        pass
        # print('Right eye is in box!')
    else:
        err_ctr += 1
        print('Right eye is NOT in box!')
    leye_box = (leye_x - margin, leye_y - margin,
                leye_x + leye_w + margin, leye_y + leye_h + margin)
    if in_box(leye_box, (leye_x_new, leye_y_new)) and in_box(leye_box, (leye_x_new + leye_w_new, leye_y_new + leye_h_new)):
        pass
        # print('Left eye is in box!')
    else:
        err_ctr += 1
        print('Left eye is NOT in box!')

    fig, ax = plt.subplots()
    ax.imshow(bw_img, cmap='gray')
    rect1 = patches.Rectangle((reye_x, reye_y), reye_w, reye_h,
                              linewidth=1, edgecolor='r', facecolor='none')
    rect2 = patches.Rectangle((leye_x, leye_y), leye_w, leye_h,
                              linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect1)
    ax.add_patch(rect2)
    rect1 = patches.Rectangle((reye_x_new, reye_y_new), reye_w_new, reye_h_new,
                              linewidth=1, edgecolor='g', facecolor='none')
    rect2 = patches.Rectangle((leye_x_new, leye_y_new), leye_w_new, leye_h_new,
                              linewidth=1, edgecolor='g', facecolor='none')
    ax.add_patch(rect1)
    ax.add_patch(rect2)
    rect1 = patches.Rectangle((reye_x - margin, reye_y - margin), reye_w + 2 * margin, reye_h + 2 * margin,
                              linewidth=1, edgecolor='b', facecolor='none')
    rect2 = patches.Rectangle((leye_x - margin, leye_y - margin), leye_w + 2 * margin, leye_h + 2 * margin,
                              linewidth=1, edgecolor='b', facecolor='none')
    ax.add_patch(rect1)
    ax.add_patch(rect2)
    plt.show()

print(f"There were {no_face} invalid faces in metadata and {err_ctr} out-of-box detected eye-landmarks.")
