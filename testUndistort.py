import os
import cv2
import numpy as np
from fisheye import unwarp
# import discorpy.losa.loadersaver as io
# import discorpy.post.postprocessing as post


# xcenter = 782.5732073914905
# ycenter = 539.8631224869504
# factors = [1.0154161165196605, -0.00012324049796004737, 1.1062875290233011e-07, -1.141523930243955e-09, 1.0148972581951193e-12]

images_path = '/Users/jibly/Documents/labHandling/lab_handling_system/temp'
images = os.listdir(images_path)  # You need to provide the paths to your calibration images here
# print(images)
dimension = None
for image in images:
    if not image.lower().endswith('.jpg'): 
        continue
    img_path = os.path.join(images_path, image)
    print(f'File: {img_path}')
    # Read the image from file
    img = cv2.imread(img_path)
    if img is None:
        continue

    cv2.imshow('undistorted', unwarp(img))
    cv2.waitKey(0)