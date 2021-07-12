import cv2
import mediapipe as mp

draw = mp.solutions.drawing_utils # Creating mediapipe drawing variable. It will be draw your face connections.

# MediaPipe Holistic consists of a new pipeline with optimized pose, face and hand components that each run in real-time.
holistic = mp.solutions.holistic 

# Creating Videocapture Class down below. Parameter 0 is your webcam on computer. If you have another camera you need to enter 1 or 2.
cap = cv2.VideoCapture(0) 

with holistic.Holistic() as hol:
    while True: # If you don't stop this program, it will be running with this loop. 
        x, camera = cap.read()

        image = cv2.cvtColor(camera, cv2.COLOR_BGR2RGB) # Specifying face connections colors.
        
        results = hol.process(image) # Processing face perception.

        draw.draw_landmarks(camera, results.face_landmarks, holistic.FACE_CONNECTIONS) # Drawing face connections.
        cv2.imshow("baver face detector", camera) # Naming program's frame here in this code.
        cv2.waitKey(2) 

cap.release()
