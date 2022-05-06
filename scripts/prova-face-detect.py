import os
import cv2
import dlib
import json
from glob import glob
import numpy as np

# First run:
#           python3 ./scripts/1a-dataset_converter_google_split.py  --dir '/home/alien/Documents/myBau/dataset/original'  --out_dir '/home/alien/Documents/myBau/dataset/google_split'

margin = 10

FACIAL_LANDMARKS_IDXS = dict([
	("mouth", (48, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 35)),
	("jaw", (0, 17))
])

p = "/home/alien/PycharmProjects/mobile-gaze-tracker/checkpoints/shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)


def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((shape.num_parts, 2), dtype=dtype)

	# loop over all facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for k in range(shape.num_parts):
		coords[k] = (shape.part(k).x, shape.part(k).y)

	# return the list of (x, y)-coordinates
	return coords


# in_dir = '/home/alien/Documents/myBau/dataset/google_split/'
in_dir = '/home/alien/Documents/myBau/dataset/mit_split/'
files = glob(os.path.join(in_dir, "*", "images", "*.jpg"))
face_col = (255, 0, 0)
for i in files:
    img = cv2.imread(i)
    bw_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    meta_file = i.replace('images', 'meta').replace('.jpg', '.json')
    with open(meta_file, 'r') as f:
        meta = json.load(f)

    leye_x, leye_y, leye_w, leye_h = meta['leye_x'], meta['leye_y'], meta['leye_w'], meta['leye_h']
    reye_x, reye_y, reye_w, reye_h = meta['reye_x'], meta['reye_y'], meta['reye_w'], meta['reye_h']
    if meta['face_valid']:
        fx, fy, fw, fh = meta['face_x'], meta['face_y'], meta['face_w'], meta['face_h']
    else:
        face_col = (0, 0, 255)
        faces = detector(bw_img, 1)
        if not faces:
            continue
        face = faces[0]
        fx, fy, fw, fh = face.left(), face.top(), face.right()-face.left(), face.bottom()-face.top()

    reye_x1, reye_y1 = reye_x, reye_y
    reye_x2, reye_y2 = reye_x + reye_w, reye_y + reye_h
    leye_x1, leye_y1 = leye_x, leye_y
    leye_x2, leye_y2 = leye_x + leye_w, leye_y + leye_h
    print(leye_w, leye_h, reye_w, reye_h)

    kps = predictor(bw_img, dlib.rectangle(fx, fy, fx + fw, fy + fh))
    kps_np = shape_to_np(kps)
    (leye_x1_new, leye_y1_new, leye_w_new, leye_h_new) = cv2.boundingRect(kps_np[slice(*FACIAL_LANDMARKS_IDXS['left_eye'], 1)])
    leye_x2_new, leye_y2_new = leye_x1_new+leye_w_new, leye_y1_new+leye_h_new
    (reye_x1_new, reye_y1_new, reye_w_new, reye_h_new) = cv2.boundingRect(kps_np[slice(*FACIAL_LANDMARKS_IDXS['right_eye'], 1)])
    reye_x2_new, reye_y2_new = reye_x1_new+reye_w_new, reye_y1_new+reye_h_new
    leye_x2_new, reye_x2_new = leye_x2_new + margin, reye_x2_new + margin
    leye_y2_new, reye_y2_new = leye_y2_new + margin, reye_y2_new + margin
    leye_x1_new, reye_x1_new = leye_x1_new - margin, reye_x1_new - margin
    leye_y1_new, reye_y1_new = leye_y1_new - margin, reye_y1_new - margin
    leye_w_new, reye_w_new = leye_w_new + 2*margin, reye_w_new + 2*margin
    leye_w_new, reye_h_new = leye_h_new + 2*margin, reye_h_new + 2*margin
    print(leye_x2_new - leye_x1_new, leye_y2_new - leye_y1_new, reye_x2_new - reye_x1_new, reye_y2_new - reye_y1_new)

    cv2.rectangle(img, (fx, fy), (fx+fw, fy+fh), face_col, 5)
    cv2.rectangle(img, (reye_x1, reye_y1), (reye_x2, reye_y2), (0, 255, 0), 1)
    cv2.rectangle(img, (leye_x1, leye_y1), (leye_x2, leye_y2), (0, 255, 0), 1)
    cv2.rectangle(img, (reye_x1_new, reye_y1_new), (reye_x2_new, reye_y2_new), (0, 255, 255), 1)
    cv2.rectangle(img, (leye_x1_new, leye_y1_new), (leye_x2_new, leye_y2_new), (0, 255, 255), 1)
    cv2.imshow('Face Detection', img)
    cv2.waitKey(200)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
