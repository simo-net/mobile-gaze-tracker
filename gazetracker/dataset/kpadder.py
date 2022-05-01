import os
import cv2
import json
import dlib
from glob import glob
from multiprocessing import Process


def in_box(box, point):
    x1, y1, x2, y2 = box
    x, y = point
    if x1 < x < x2 and y1 < y < y2:
        return True
    return False


def keypoints_adder(files: list, p: str):
    """
    Add eye landmarks to the metafiles of the Gaze Capture Dataset using DLib.
    """

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)

    buffer = 10
    no_face, err_ctr = 0, 0
    for i in files:
        img = cv2.imread(i)
        bw_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        meta_file = i.replace('images', 'meta').replace('.jpg', '.json')
        with open(meta_file, 'r') as f:
            meta = json.load(f)

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
        face_rect = dlib.rectangle(fx, fy, fx+fw, fy+fh)
        kps = predictor(bw_img, face_rect)

        # make sure the detected landmark points are within the eye bounding box (with margin "buffer" in all directions)
        eye_box = (reye_x - buffer,
                   reye_y - buffer,
                   reye_x + reye_w + buffer,
                   reye_y + reye_h + buffer)
        if in_box(eye_box, (kps.part(36).x, kps.part(36).y) and in_box(eye_box, (kps.part(39).x, kps.part(39).y))):
            meta['reye_x1'], meta['reye_y1'] = kps.part(36).x, kps.part(36).y
            meta['reye_x2'], meta['reye_y2'] = kps.part(39).x, kps.part(39).y
        else:
            err_ctr += 1
            # # TODO: check this! Doing so reye_y1 is equal to reye_y2
            # meta['reye_x1'], meta['reye_y1'] = reye_x, reye_y + (reye_h // 2)
            # meta['reye_x2'], meta['reye_y2'] = reye_x + reye_w, reye_y + (reye_h // 2)
            # # TODO: maybe the following is better?
            meta['reye_x1'], meta['reye_y1'] = reye_x, reye_y
            meta['reye_x2'], meta['reye_y2'] = reye_x + reye_w, reye_y + reye_h

        if in_box(eye_box, (kps.part(42).x, kps.part(42).y) and in_box(eye_box, (kps.part(45).x, kps.part(45).y))):
            meta['leye_x1'], meta['leye_y1'] = kps.part(42).x, kps.part(42).y
            meta['leye_x2'], meta['leye_y2'] = kps.part(45).x, kps.part(45).y
        else:
            err_ctr += 1
            # # TODO: check this! Doing so leye_y1 is equal to leye_y2
            # meta['leye_x1'], meta['leye_y1'] = leye_x, leye_y - (leye_h // 2)
            # meta['leye_x2'], meta['leye_y2'] = leye_x + leye_w, leye_y + (leye_h // 2)
            # # TODO: maybe the following is better?
            meta['leye_x1'], meta['leye_y1'] = leye_x, leye_y
            meta['leye_x2'], meta['leye_y2'] = leye_x + leye_w, leye_y + leye_h

        with open(meta_file, 'w') as outfile:
            json.dump(meta, outfile)

    print(f"there were {no_face} face errors and {err_ctr} out-of-box eye-landmarks errors.")


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
