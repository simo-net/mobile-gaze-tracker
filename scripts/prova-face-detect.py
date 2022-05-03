import os
import cv2
import dlib
import json
from glob import glob

# First run:
#           python3 ./scripts/1a-dataset_converter_google_split.py  --dir '/home/alien/Documents/myBau/dataset/original'  --out_dir '/home/alien/Documents/myBau/dataset/google_split'

p = "/home/alien/PycharmProjects/mobile-gaze-tracker/checkpoints/shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

# in_dir = '/home/alien/Documents/myBau/dataset/google_split/'
in_dir = '/home/alien/Documents/myBau/dataset/mit_split/'
files = glob(os.path.join(in_dir, "*", "images", "*.jpg"))
for i in files:
    face_col = (255, 0, 0)

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
        fx, fy, fw, fh = face.left(), face.top(), face.right() - face.left(), face.bottom() - face.top()

    reye_x1, reye_y1 = reye_x, reye_y
    reye_x2, reye_y2 = reye_x + reye_w, reye_y + reye_h
    leye_x1, leye_y1 = leye_x, leye_y
    leye_x2, leye_y2 = leye_x + leye_w, leye_y + leye_h
    print(leye_w, leye_h, reye_w, reye_h)

    kps = predictor(bw_img, dlib.rectangle(fx, fy, fx + fw, fy + fh))
    reye_x1_raw, reye_y1_raw = kps.part(36).x, kps.part(36).y
    reye_x2_raw, reye_y2_raw = kps.part(39).x, kps.part(39).y
    leye_x1_raw, leye_y1_raw = kps.part(42).x, kps.part(42).y
    leye_x2_raw, leye_y2_raw = kps.part(45).x, kps.part(45).y

    reye_x1_new, reye_y1_new = reye_x1_raw+abs(reye_x2_raw-reye_x1_raw)//2-reye_w//2, reye_y1_raw+abs(reye_y2_raw-reye_y1_raw)//2-reye_h//2
    reye_x2_new, reye_y2_new = reye_x1_new+reye_w, reye_y1_new+reye_h
    leye_x1_new, leye_y1_new = leye_x1_raw+abs(leye_x2_raw-leye_x1_raw)//2-leye_w//2, leye_y1_raw+abs(leye_y2_raw-leye_y1_raw)//2-leye_h//2
    leye_x2_new, leye_y2_new = leye_x1_new+leye_w, leye_y1_new+leye_h
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
