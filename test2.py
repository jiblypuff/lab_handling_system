import yolo
import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()
while True:
    cx, cy, dist = yolo.getClosestObject(cap)
    print(cx, cy, dist)