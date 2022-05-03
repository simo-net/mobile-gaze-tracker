import cv2
import dlib

p = "/home/alien/PycharmProjects/mobile-gaze-tracker/checkpoints/shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

capture = cv2.VideoCapture(0)

while True:
    response, img = capture.read()
    if not response:
        continue
    # bw_img = cv2.flip(img, 1)
    bw_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detector(bw_img, 1)
    if not faces:
        continue
    face = faces[0]
    fx, fy, fw, fh = face.left(), face.top(), face.right() - face.left(), face.bottom() - face.top()

    face_rect = dlib.rectangle(fx, fy, fx + fw, fy + fh)
    kps = predictor(bw_img, face_rect)
    reye_x1, reye_y1 = kps.part(36).x, kps.part(36).y
    reye_x2, reye_y2 = kps.part(39).x, kps.part(39).y
    leye_x1, leye_y1 = kps.part(42).x, kps.part(42).y
    leye_x2, leye_y2 = kps.part(45).x, kps.part(45).y

    cv2.rectangle(img, (fx, fy), (fx+fw, fy+fh), (0, 0, 255), 5)
    cv2.rectangle(img, (reye_x1, reye_y1), (reye_x2, reye_y2), (0, 255, 0), 3)
    cv2.rectangle(img, (leye_x1, leye_y1), (leye_x2, leye_y2), (0, 255, 0), 3)
    cv2.imshow('Face Detection', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
capture.release()
