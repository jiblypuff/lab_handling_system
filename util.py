import cv2
from ultralytics import YOLOv10
import supervision as sv
import numpy as np
import matplotlib.pyplot as plt
import discorpy.post.postprocessing as post
import os
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# fisheye parameters
xcenter = 785.3054074949058
ycenter = 530.4032340121979
factors = [ 1.00851460e+00, -1.63425112e-05, -3.77548357e-07, -7.29162454e-11, 1.96865934e-13]

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cpu" # change to cuda if using a device with gpu

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)

def unwarp(img):
    img_corrected = np.copy(img)
    for i in range(img.shape[-1]):
        img_corrected[:, :, i] = post.unwarp_image_backward(img[:, :, i], xcenter,
                                                            ycenter, factors)
    return img_corrected


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img, aspect='auto')

def segment(img):
    # print(img.shape)
    w, h = img.shape[:2]
    masks = mask_generator.generate(img)
    # h, w = img.shape[:2]
    fig, ax = plt.subplots()
    # fig.set_size_inches(w, h)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_frame_on(False)
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img, aspect='auto')
    show_anns(masks)
    # ax.axis('off')
    # plt.box(on=None)
    fig.canvas.draw()
    return cv2.resize(cv2.cvtColor(np.array(fig.canvas.renderer.buffer_rgba()), cv2.COLOR_RGBA2BGR), (h,w))

def img_to_polygons(image):
    cordnt_list = []
    centers_list = []
    areas_list = []
    rec_list = []
    
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    frame = clahe.apply(frame)

    frame = cv2.bilateralFilter(frame,5,75,75)

    high_thresh, thresh_im = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lowThresh = 0.5*high_thresh

    frame = cv2.Canny(frame, 50, 150)

    frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 4)

    out = 'output/edges.jpg'

    cv2.imwrite(out, frame)

    contours, hierarchy = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return cordnt_list, centers_list

    contour_areas = [cv2.contourArea(contour) for contour in contours]
    max_area = max(contour_areas)

    for contour in contours:
        area = cv2.contourArea(contour)

        if area > 300:
            coordinates = contour.squeeze().astype(float).tolist()
            if len(coordinates) > 2:
                M = cv2.moments(contour)
                cX = int(M["m10"] / (M["m00"] + 1e-5))
                cY = int(M["m01"] / (M["m00"] + 1e-5))
                
                crdnts = [{'x': i[0], 'y': i[1]} for i in coordinates]
                cordnt_list.append(crdnts)
                centers_list.append({'x': cX, 'y': cY})
                areas_list.append(area)
                    

    return cordnt_list, centers_list, areas_list

def drawContours(img, polys, centers, areas, offset):
    biggest = np.argmax(areas)
    print('biggest ', biggest)

    ply_list = [[pnt['x'], pnt['y']] for pnt in polys[biggest]]

    # Define the vertices of the polygon
    vertices = np.array(ply_list, dtype=np.int32)
    vertices += offset

    cX = centers[biggest]['x'] + offset[0]
    cY = centers[biggest]['y'] + offset[1]

    cv2.polylines(img, [vertices], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.circle(img, (cX, cY), 12, (0, 0, 255), -1)
    cv2.putText(img, "CONTOUR CENTER", (cX - 20, cY - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)