import cv2
from ultralytics import YOLOv10
import supervision as sv
import numpy as np
import matplotlib.pyplot as plt
from fisheye import unwarp
from segment import segment

# Load the YOLOv10 model
model = YOLOv10('/Users/jibly/Documents/labHandling/lab_handling_system/models/best.pt')
transparent = YOLOv10('/Users/jibly/Documents/labHandling/lab_handling_system/models/transparent.pt')

# cap = cv2.VideoCapture('/Users/jibly/Documents/labHandling/lab_handling_system/videos/vid2.mov')
# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print("Error: Could not open video stream.")
#     exit()

bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

output_video_path = '/Users/jibly/Documents/labHandling/lab_handling_system/output/pred.mp4'

# mtx = np.array([[906.55114597,  -2.30814081,     719.28355776],
#                 [0.,            902.56494318,   477.55878025],
#                 [0.,            0.,           1.        ]])
# dist = np.array([[-0.46650992],
#                 [ 1.88436159],
#                 [-0.76346618],
#                 [-0.87372484]])



goal_x = 1001
goal_y = 310




# Define the codec and create a VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' for .avi format
# out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# return center and distance of object closest to center of frame
def getClosestObject(cap, trans=False):
# while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame.")
        # break
        return -1, -1, -1
    
    #----TODO
    # undisotort the image before pasing thru yolo
    cv2.imwrite('/Users/jibly/Documents/labHandling/lab_handling_system/output/original.jpg', frame)
    frame = unwarp(frame)

    results = model(frame, conf=0.3)[0]
    
    detections = sv.Detections.from_ultralytics(results)

    annotated_image = bounding_box_annotator.annotate(
        scene=frame, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections)
    
    if trans:
        masked = segment(frame)
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
        # print(box)
        cx = (box[0][2] + box[0][0]) / 2 
        cy = (box[0][3] + box[0][1]) / 2

        cur_dist = np.sqrt((cx - goal_x)**2 + (cy-goal_y)**2)
        if(cur_dist < min_dist):
            min_dist = cur_dist
            closest_cent = (int(cx),int(cy))
            conf = box[2]
    
    cv2.circle(annotated_image, (goal_x, goal_y), 12, (255, 0, 0), -1)
    cv2.putText(annotated_image, "goal", (goal_x - 20, goal_y - 20), 
    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.circle(annotated_image, closest_cent, 7, (255, 255, 255), -1)
    cv2.putText(annotated_image, "conf="+str(conf), (int(closest_cent[0]) - 20, int(closest_cent[1]) - 20), 
    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    cv2.imwrite('/Users/jibly/Documents/labHandling/lab_handling_system/output/stream.jpg', annotated_image)
    
    return closest_cent[0], closest_cent[1], min_dist
    # # Exit if 'q' is pressed
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     return

def getTransparentObject(cap):
# while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame.")
        # break
        return -1, -1, -1
    
    #----TODO
    # undisotort the image before pasing thru yolo
    cv2.imwrite('/Users/jibly/Documents/labHandling/lab_handling_system/output/original.jpg', frame)
    frame = unwarp(frame)
    unmasked = frame.copy()

    frame = segment(frame)

    results = transparent(frame, conf=0.3)[0]
    
    detections = sv.Detections.from_ultralytics(results)

    annotated_image = bounding_box_annotator.annotate(
        scene=unmasked, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections)
    
    # exit if no objects are detected
    if(len(detections) == 0): 
        print("no detections")
        return -1, -1, -1

    img_center = np.asarray(annotated_image.shape[:2]) / 2
    # print(img_center)
    cv2.circle(annotated_image, (goal_x, goal_y), 12, (255, 0, 0), -1)
    cv2.putText(annotated_image, "goal", (goal_x - 20, goal_y - 20), 
    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    min_dist = float('inf')
    closest_cent = (-1, -1)
    conf = -1
    for box in detections:
        # print(box)
        cx = (box[0][2] + box[0][0]) / 2 
        cy = (box[0][3] + box[0][1]) / 2

        cur_dist = np.sqrt((cx - goal_x)**2 + (cy-goal_y)**2)
        if(cur_dist < min_dist):
            min_dist = cur_dist
            closest_cent = (int(cx),int(cy))
            conf = box[2]

    cv2.circle(annotated_image, closest_cent, 7, (255, 255, 255), -1)
    cv2.putText(annotated_image, "conf="+str(conf), (int(closest_cent[0]) - 20, int(closest_cent[1]) - 20), 
    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    cv2.imwrite('/Users/jibly/Documents/labHandling/lab_handling_system/output/transparent.jpg', annotated_image)
    

# Release the video stream and close windows
# cap.release()
# out.release()
# cv2.destroyAllWindows()
    
