import cv2
import numpy as np
import os


def img_to_polygons(image):
    cordnt_list = []
    centers_list = []
    areas_list = []
    rec_list = []
    
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    frame = clahe.apply(frame)

    # frame = cv2.GaussianBlur(frame, (21,21), 0)
    frame = cv2.bilateralFilter(frame,5,75,75)

    high_thresh, thresh_im = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lowThresh = 0.5*high_thresh

    frame = cv2.Canny(frame, 50, 150)

    frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # frame = cv2.bilateralFilter(frame,5,150,150)

    # high_thresh, thresh_im = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # lowThresh = 0.5*high_thresh

    # frame = cv2.Canny(frame, 50, 150)

    out = os.path.join(outPut_dir, "edges.jpg")

    cv2.imwrite(out, frame)

    contours, hierarchy = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return cordnt_list, centers_list

    contour_areas = [cv2.contourArea(contour) for contour in contours]
    max_area = max(contour_areas)

    # Extract coordinates of white regions
    for contour in contours:
        # epsilon = 0.5*cv2.arcLength(contour,True)
        # contour = cv2.approxPolyDP(contour,epsilon,True)
        area = cv2.contourArea(contour)

        # Filter based on the area conditions
        if area > 300:
            coordinates = contour.squeeze().astype(float).tolist()
            if len(coordinates) > 2:
                M = cv2.moments(contour)
                cX = int(M["m10"] / (M["m00"] + 1e-5))
                cY = int(M["m01"] / (M["m00"] + 1e-5))
                
                # crdnts = [{'x': i[0], 'y': i[1]} for i in coordinates]
                # rect = {}
                # for i in cv2.boundingRect(contour):
                #     rect = {'x': i[0], 'y': i[1], 'w': i[2], 'h': i[3]} 
                # cordnt_list.append(crdnts)
                rec_list.append(cv2.boundingRect(contour))
                centers_list.append({'x': cX, 'y': cY})
                areas_list.append(area)
                    

    # tup = (len(cordnt_list), len(centers_list), len(areas_list))
    # print(tup)
    return rec_list, centers_list, areas_list

count = 0
center_X = 0
center_Y = 0
path = os.getcwd()

# Define input and output directories
outPut_dir = os.path.join(path, 'output')
os.makedirs(outPut_dir, exist_ok=True)

vid = cv2.VideoCapture('videos/vid3.mov')

while(True):

    ret, frame = vid.read()

    if not ret:
        print("video ended")
        break

    scale_percent = 100
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)

    frame = cv2.resize(frame, dim , interpolation = cv2.INTER_AREA)

    frame_copy = frame.copy()

    # Input and output file paths
    fout = os.path.join(outPut_dir, "stream.jpg")

    # Extract polygons from the image
    rect, centers, areas = img_to_polygons(frame_copy)

    # Get the height and width of the image
    height, width = frame_copy.shape[:2]

    # Create a black image with the same dimensions as the input
    # black_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Draw polygons on the black image
    for i, rec in enumerate(rect):
        # ply_list = [[pnt['x'], pnt['y']] for pnt in ply]

        # Define the vertices of the polygon
        # vertices = np.array(ply_list, dtype=np.int32)

        cX = centers[i]['x']
        cY = centers[i]['y']
        # Draw the polygon on the black image
        # cv2.polylines(frame_copy, [vertices], isClosed=True, color=(0, 255, 0), thickness=2)
        x,y,w,h = rec
        ar = w*h
        if ar > 3200 and ar < 300000:
            cv2.rectangle(frame_copy,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.circle(frame_copy, (cX, cY), 7, (255, 255, 255), -1)
            cv2.putText(frame_copy, "area="+str(ar), (cX - 20, cY - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Save the output image
    cv2.imwrite(fout, frame_copy)

    input()