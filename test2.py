import yoloLib
import cv2
from segment import segment

cap = cv2.VideoCapture('/Users/jibly/Documents/labHandling/lab_handling_system/videos/vid3.mov')
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()
while True:
    # ret, frame = cap.read()
    # seg = segment(frame)
    # cv2.imshow('test', seg)
    # cv2.waitKey(0)
    # yolo.getTransparentObject(cap)
    cx, cy, dist = yoloLib.getClosestObject(cap, True)
    print(cx, cy, dist)