import cv2
import dlib
import numpy as np
from imutils import face_utils

FACIAL_LANDMARKS_IDXS = dict([
	("mouth", (48, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 35)),
	("jaw", (0, 17))
])
buffer = 10

p_hog = "/home/alien/PycharmProjects/mobile-gaze-tracker/checkpoints/shape_predictor_68_face_landmarks.dat"
# p_hog = "/home/alien/PycharmProjects/mobile-gaze-tracker/checkpoints/shape_predictor_68_face_landmarks_GTX.dat"
p_cnn = "/home/alien/PycharmProjects/mobile-gaze-tracker/checkpoints/face_detector_mmod.dat"

# face_detector = dlib.get_frontal_face_detector()
face_detector = dlib.cnn_face_detection_model_v1(p_cnn)
kps_predictor = dlib.shape_predictor(p_hog)


def process_boxes(box):
    xmin = box.rect.left()
    ymin = box.rect.top()
    xmax = box.rect.right()
    ymax = box.rect.bottom()
    return [int(xmin), int(ymin), int(xmax), int(ymax)]


capture = cv2.VideoCapture(0)

while True:
    response, img = capture.read()
    if not response:
        continue

    bw_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = face_detector(bw_img, 1)
    if not rects:
        continue
    face = rects[0]
    fx, fy, fw, fh = face.left(), face.top(), face.right() - face.left(), face.bottom() - face.top()

    # rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # rects = face_detector(rgb_img, 1)
    # if not rects:
    #     continue
    # for box in rects:
    #     res_box = process_boxes(box)
    #     cv2.rectangle(img,
    #                   (res_box[0], res_box[1]),
    #                   (res_box[2], res_box[3]),
    #                   (0, 255, 0), 2)
    # face = rects[0]
    # fx, fy, fw, fh = face[0], face[1], face[2]-face[0], face[3]-face[1]

    face_rect = dlib.rectangle(fx, fy, fx + fw, fy + fh)
    kps = kps_predictor(bw_img, face_rect)

    # reye_x1, reye_y1 = kps.part(36).x, kps.part(36).y
    # reye_x2, reye_y2 = kps.part(39).x, kps.part(39).y
    # leye_x1, leye_y1 = kps.part(42).x, kps.part(42).y
    # leye_x2, leye_y2 = kps.part(45).x, kps.part(45).y
    # cv2.rectangle(img, (fx, fy), (fx+fw, fy+fh), (0, 0, 255), 5)
    # cv2.rectangle(img, (reye_x1, reye_y1), (reye_x2, reye_y2), (0, 255, 0), 3)
    # cv2.rectangle(img, (leye_x1, leye_y1), (leye_x2, leye_y2), (0, 255, 0), 3)
    # cv2.imshow('Face Detection', img)

    shape = face_utils.shape_to_np(kps)
    # for (name, (i, j)) in FACIAL_LANDMARKS_IDXS.items():
        # for (x, y) in shape[i:j]:
        #     cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
    for (i, j) in [FACIAL_LANDMARKS_IDXS['right_eye'], FACIAL_LANDMARKS_IDXS['left_eye']]:
        (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
        cv2.rectangle(img, (x-buffer, y-buffer), (x+w+buffer, y+h+buffer), (0, 255, 0), 3)

    cv2.imshow('Face Detection', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
capture.release()
