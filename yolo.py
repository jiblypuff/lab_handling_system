import cv2
from ultralytics import YOLOv10
import supervision as sv
import numpy as np
import matplotlib.pyplot as plt
import util

# Load the YOLOv10 model
model = YOLOv10('/Users/jibly/Documents/labHandling/lab_handling_system/models/best.pt')
transparent = YOLOv10('/Users/jibly/Documents/labHandling/lab_handling_system/models/transparent.pt')

bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

goal_x = 1001
goal_y = 310


# return center and distance of object closest to center of frame
def getClosestObject(cap, trans=False):
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame.")
        # break
        return -1, -1, -1
    
    # undisotort the image before pasing thru yolo
    cv2.imwrite('/Users/jibly/Documents/labHandling/lab_handling_system/output/original.jpg', frame)
    frame = util.unwarp(frame)

    results = model(frame, conf=0.3)[0]
    
    detections = sv.Detections.from_ultralytics(results)

    annotated_image = bounding_box_annotator.annotate(
        scene=frame, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections)
    
    if trans:
        masked = util.segment(frame)
        transRes = transparent(masked, conf=0.3)[0]
        transDetections = sv.Detections.from_ultralytics(transRes)

        annotated_image = bounding_box_annotator.annotate(
            scene=annotated_image, detections=transDetections)
        annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=transDetections)
        detections = sv.Detections.merge([detections, transDetections])

    
    # exit if no objects are detected
    if(len(detections) == 0): 
        print("no detections")
        cv2.imwrite('/Users/jibly/Documents/labHandling/lab_handling_system/output/stream.jpg', frame)
        return -1, -1, -1

    min_dist = float('inf')
    closest_cent = (-1, -1)
    conf = -1
    for box in detections:
        x1, y1, x2, y2 = box[0]
        cx = (x1 + x2) / 2 
        cy = (y1 + y2) / 2

        label = box[5]['class_name']

        # if label == 'scissors' or label == 'screwdriver':
        #     x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        #     polygons, centers, areas = util.img_to_polygons(frame[y1:y2, x1:x2, :])
        #     util.drawContours(annotated_image, polygons, centers, areas, (x1,y1))

        cur_dist = np.sqrt((cx - goal_x)**2 + (cy-goal_y)**2)
        if(cur_dist < min_dist):
            min_dist = cur_dist 
            closest_cent = (int(cx),int(cy))
            conf = box[2]
    
    # cv2.circle(annotated_image, (goal_x, goal_y), 12, (255, 0, 0), -1)
    # cv2.putText(annotated_image, "goal", (goal_x - 20, goal_y - 20), 
    # cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.circle(annotated_image, closest_cent, 7, (255, 0, 0), -1)
    # cv2.putText(annotated_image, "conf="+str(conf), (int(closest_cent[0]) - 20, int(closest_cent[1]) - 20), 
    # cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    cv2.imwrite('/Users/jibly/Documents/labHandling/lab_handling_system/output/stream.jpg', annotated_image)
    
    return closest_cent[0], closest_cent[1], min_dist
    