import cv2
import numpy as np

count = 0
center_X = 0
center_Y = 0
while(True):
    vid = cv2.VideoCapture(0)
    ret, frame = vid.read()

    # print("Object found: " + str(object_found))

    # print('Original Dimensions: ', frame.shape)

    scale_percent = 50
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)

    # print(dim)

    # cv2.imwrite("before"+str(count)+".jpg", frame)
    # cv2.waitKey(0)

    frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

    frame_copy = frame.copy()



     # converting to LAB color space
    lab= cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    frame = cv2.merge((cl,a,b))

    # Converting image from LAB Color model to BGR color spcae
    frame = cv2.cvtColor(frame, cv2.COLOR_LAB2BGR)

    # cv2.imshow("enhanced", frame)
    # cv2.waitKey(0)

    frame = cv2.GaussianBlur(frame, (21,21), 2)

    # cv2.imwrite("filtered_enhanced"+str(count)+".jpg", frame)

    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # cv2.imshow("gray", thresh)
    # cv2.waitKey(0)

    contours, hiearchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # frame_copy = frame.copy()

    areas = []
    centers = []

    for c in contours:
        # find the area of the contour
        areas.append(cv2.contourArea(c))
        # compute the center of the contour
        M = cv2.moments(c)
        cX = int(M["m10"] / (M["m00"] + 1e-5))
        cY = int(M["m01"] / (M["m00"] + 1e-5))

        centers.append((cX, cY))

    all_conts = {}

    for i in range(len(areas)):
        all_conts[i] = {'area' : areas[i], 'center' : centers[i]}

    sorted_conts = sorted(all_conts.items(), key = lambda x:x[1]["area"], reverse = True)

    # print(sorted_conts)

    area_num = 0

    conts_final = []

    if(len(areas) != 0):
        for i in range(len(areas)):
            if(sorted_conts[i][1]['area'] * 1000 > sorted_conts[0][1]['area']
               and sorted_conts[i][1]['area'] * 2 < sorted_conts[0][1]['area']):
                center_X = sorted_conts[i][1]['center'][0]
                center_Y = sorted_conts[i][1]['center'][1]
                conts_final.append(contours[sorted_conts[i][0]])
                area_num = area_num + 1


    cv2.drawContours(frame_copy, conts_final, -1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("frame"+str(count), frame_copy)
    cv2.waitKey(0);
    # cv2.imwrite("test"+str(count)+".jpg", frame_copy)
    # cv2.waitKey(0)


    print(count)
    # break
    # cv2.imshow("contour"+str(count)+".jpg", frame_copy)
    # cv2.waitKey(0)
    # print("CenterX: " + str(center_X))
    # print("CenterY: " + str(center_Y))

    count += 1