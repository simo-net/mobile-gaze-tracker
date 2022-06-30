import os
import cv2
import json
import dlib
import random
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches

W, H = 128, 128
num_samples = -1
in_dir = '/home/alien/Documents/myBau/dataset/google_split'
# in_dir = '/home/simonetunige/data/mobile-gaze/gaze-capture_google-split'
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


def find_center(box):
    (x1, y1), (x2, y2) = box
    w, h = x2-x1, y2-y1
    return x1 + w//2, y1 + h//2


def in_box(box, point):
    (x1, y1), (x2, y2) = box
    x, y = point
    if x1 < x < x2 and y1 < y < y2:
        return True
    return False


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


def kps_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)
    # loop over all facial landmarks and convert them to a 2-tuple of (x, y)-coordinates
    for k in range(shape.num_parts):
        coords[k] = (shape.part(k).x, shape.part(k).y)
    # return the list of (x, y)-coordinates
    return coords


def box_with_margin(box, margins):
    (x1, y1), (x2, y2) = box
    w, h = x2 - x1, y2 - y1
    x_margin, y_margin = margins
    w_new, h_new = w + 2 * x_margin, h + 2 * y_margin
    x1_new, y1_new = x1 - x_margin, y1 - y_margin
    x2_new, y2_new = x1_new + w_new, y1_new + h_new
    return [(x1_new, y1_new), (x2_new, y2_new)]


def main():

    margin = 30
    no_face, err_eye, err_img = 0, 0, 0
    for img_file in files:
        print('*'*30)

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

        # Detect face if no valid face info is present in the metadata and check reference eye-landmarks are in the face box
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

        # Resize the new eye ROIs to the target shape
        reye_w, reye_h = W, H
        (reye_x1, reye_y1), (reye_x2, reye_y2) = box_from_center(
            center=find_center(box=[(reye_x1, reye_y1), (reye_x2, reye_y2)]),
            box_shape=(W, H), img_shape=(w_img, h_img))
        leye_w, leye_h = W, H
        (leye_x1, leye_y1), (leye_x2, leye_y2) = box_from_center(
            center=find_center(box=[(leye_x1, leye_y1), (leye_x2, leye_y2)]),
            box_shape=(W, H), img_shape=(w_img, h_img))

        # Resize the reference eye ROIs to the target shape
        reye_w_ref, reye_h_ref = W, H
        (reye_x1_ref, reye_y1_ref), (reye_x2_ref, reye_y2_ref) = box_from_center(
            center=find_center(box=[(reye_x1_ref, reye_y1_ref), (reye_x2_ref, reye_y2_ref)]),
            box_shape=(W, H), img_shape=(w_img, h_img))
        leye_w_ref, leye_h_ref = W, H
        (leye_x1_ref, leye_y1_ref), (leye_x2_ref, leye_y2_ref) = box_from_center(
            center=find_center(box=[(leye_x1_ref, leye_y1_ref), (leye_x2_ref, leye_y2_ref)]),
            box_shape=(W, H), img_shape=(w_img, h_img))

        # Make sure the detected landmark points are within the image and the original eye bounding boxes (from metadata)
        img_box = [(0, 0), (w_img, h_img)]
        if in_box(img_box, (reye_x1, reye_y1)) and in_box(img_box, (reye_x2, reye_y2)):
            # print('Right eye is inside the image!')
            reye_box = [(reye_x1_ref - margin, reye_y1_ref - margin), (reye_x2_ref + margin, reye_y2_ref + margin)]
            if in_box(reye_box, (reye_x1, reye_y1)) and in_box(reye_box, (reye_x2, reye_y2)):
                pass
                # print('Right eye is in box!')
            else:
                err_eye += 1
                print(f'Right eye {img_file.split("/")[-1][:-4]} is NOT in box!')
            # pass
        else:
            err_img += 1
            print(f'Right eye is NOT inside the image!')
        if in_box(img_box, (leye_x1, leye_y1)) and in_box(img_box, (leye_x2, leye_y2)):
            # print('Left eye is inside the image!')
            leye_box = [(leye_x1_ref - margin, leye_y1_ref - margin), (leye_x2_ref + margin, leye_y2_ref + margin)]
            if in_box(leye_box, (leye_x1, leye_y1)) and in_box(leye_box, (leye_x2, leye_y2)):
                pass
                # print('Left eye is in box!')
            else:
                err_eye += 1
                print(f'Left eye {img_file.split("/")[-1][:-4]} is NOT in box!')
            # pass
        else:
            err_img += 1
            print(f'Left eye is NOT inside the image!')

        fig, ax = plt.subplots()
        ax.imshow(bw_img, cmap='gray')
        rect1 = patches.Rectangle((reye_x1_ref - margin, reye_y1_ref - margin), reye_w_ref + 2 * margin, reye_h_ref + 2 * margin,
                                  linewidth=1, linestyle='--', edgecolor='r', facecolor='none')
        rect2 = patches.Rectangle((leye_x1_ref - margin, leye_y1_ref - margin), leye_w_ref + 2 * margin, leye_h_ref + 2 * margin,
                                  linewidth=1, linestyle='--', edgecolor='r', facecolor='none')
        ax.add_patch(rect1)
        ax.add_patch(rect2)
        # rect1 = patches.Rectangle((reye_x1_ref, reye_y1_ref), reye_w_ref, reye_h_ref,
        #                           linewidth=1, edgecolor='r', facecolor='none')
        # rect2 = patches.Rectangle((leye_x1_ref, leye_y1_ref), leye_w_ref, leye_h_ref,
        #                           linewidth=1, edgecolor='r', facecolor='none')
        # ax.add_patch(rect1)
        # ax.add_patch(rect2)
        rect1 = patches.Rectangle((reye_x1, reye_y1), reye_w, reye_h,
                                  linewidth=1, edgecolor='g', facecolor='none')
        rect2 = patches.Rectangle((leye_x1, leye_y1), leye_w, leye_h,
                                  linewidth=1, edgecolor='g', facecolor='none')
        ax.add_patch(rect1)
        ax.add_patch(rect2)
        plt.show()

    print(f"There were {no_face} invalid faces in metadata and {err_img} out-of-image and {err_eye} out-of-eye detected eye-landmarks.")


if __name__ == "__main__":
    main()
