

import cv2
import mediapipe as mp

draw = mp.solutions.drawing_utils
holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)

with holistic.Holistic() as hol:
    while True:
        x, camera = cap.read()

        image = cv2.cvtColor(camera, cv2.COLOR_BGR2RGB)
        results = hol.process(image)

        draw.draw_landmarks(camera, results.face_landmarks, holistic.FACE_CONNECTIONS)
        cv2.imshow("baver face detector", camera)
        cv2.waitKey(2)

cap.release()