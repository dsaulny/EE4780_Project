import cv2
import mediapipe as mp
#import csv
#import os
#import numpy as np


mp_holistic=mp.solutions.holistic
mp_drawing=mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7)

hand_circle = mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=6)
hand_line = mp_drawing.DrawingSpec(color=(255,0,255), thickness=2, circle_radius=6)

pose_circle = mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=8)
pose_line = mp_drawing.DrawingSpec(color=(255,255,0), thickness=2, circle_radius=8)

face_line = mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=4)
face_circle = mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=4)

cam=cv2.VideoCapture(0)
while cam.isOpened():
    data, img = cam.read()
    cv2.imshow('Camera', img)

    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=holistic.process(img)

    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

    mp_drawing.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              hand_line, hand_circle)
    mp_drawing.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              hand_line, hand_circle)
    # mp_drawing.draw_landmarks(img, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                              # face_line, face_circle)
    mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              pose_line, pose_circle)

    cv2.imshow('Detection', img)

    if cv2.waitKey(5) & 0xFF == ord("q"):
        break



cam.release()
cv2.destroyAllWindows()