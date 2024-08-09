import cv2
from ultralytics import YOLOv10
import supervision as sv
import numpy as np
import matplotlib.pyplot as plt
from fisheye import unwarp
from PIL import Image
import torch
import sys
sys.path.append("..")
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor

# Load the YOLOv10 model
#model = YOLOv10('/Users/jibly/Documents/labHandling/lab_handling_system/models/best.pt')
model = YOLOv10('/Users/y.i_2.0/Documents/course/lab_handling_system/models/best.pt')

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
#device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
#sam.to(device=device)
predictor = SamPredictor(sam)

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
def getClosestObject(cap):
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

    # exit if no objects are detected
    if(len(detections) == 0): 
        print("no detections")
        return -1, -1, -1
    
    annotated_image = bounding_box_annotator.annotate(
        scene=frame, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections)
    
    
    img_center = np.asarray(annotated_image.shape[:2]) / 2
    # print(img_center)
    cv2.circle(annotated_image, (goal_x, goal_y), 12, (255, 0, 0), -1)
    cv2.putText(annotated_image, "goal", (goal_x - 20, goal_y - 20), 
    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    min_dist = float('inf')
    closest_cent = (-1, -1)
    conf = -1
    closest_box = None
    closest_label = None

    for i, box in enumerate(detections):
        # print(box)
        cx = (box[0][2] + box[0][0]) / 2 
        cy = (box[0][3] + box[0][1]) / 2

        cur_dist = np.sqrt((cx - goal_x)**2 + (cy-goal_y)**2)
        if(cur_dist < min_dist):
            min_dist = cur_dist
            closest_cent = (int(cx),int(cy))
            conf = box[2]
            closest_box = box
            closest_label = detections.names[int(detections.cls[i])]

    cv2.circle(annotated_image, closest_cent, 7, (255, 255, 255), -1)
    cv2.putText(annotated_image, "conf="+str(conf), (int(closest_cent[0]) - 20, int(closest_cent[1]) - 20), 
    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    #cv2.imwrite('/Users/jibly/Documents/labHandling/lab_handling_system/output/stream.jpg', annotated_image)
    cv2.imwrite('/Users/y.i_2.0/Documents/course/lab_handling_system/output/stream.jpg', annotated_image)
    
    if closest_label != "scissors":
        return closest_cent[0], closest_cent[1], min_dist
    
    # If the closest object is labeled as "scissors", use SAM model to generate the mask
    predictor.set_image(frame)
    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=closest_box[None, :],
        multimask_output=False,
    )
    #view the masked image
    # plt.figure(figsize=(10, 10))
    # plt.imshow(image)
    # show_mask(masks[0], plt.gca())
    # show_box(input_box, plt.gca())
    # plt.axis('off')
    # plt.show()

    # Get the mask and calculate its center
    mask = masks[0]
    y_coords, x_coords = np.where(mask)
    if len(x_coords) > 0 and len(y_coords) > 0:
        center_x = int(np.mean(x_coords))
        center_y = int(np.mean(y_coords))

        # Annotate the center of the SAM mask on the image
        cv2.circle(annotated_image, (center_x, center_y), 5, (0, 255, 0), -1)
        cv2.putText(annotated_image, "SAM Center", (center_x - 20, center_y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save the annotated frame
        #cv2.imwrite('/Users/jibly/Documents/labHandling/lab_handling_system/output/stream_with_sam.jpg', annotated_image)
        cv2.imwrite('/Users/y.i_2.0/Documents/course/lab_handling_system/output/stream_with_sam.jpg', annotated_image)

        return center_x, center_y, np.sqrt((center_x - goal_x) ** 2 + (center_y - goal_y) ** 2)
    

    return closest_cent[0], closest_cent[1], min_dist
    # # Exit if 'q' is pressed
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     return

# Release the video stream and close windows
# cap.release()
# out.release()
# cv2.destroyAllWindows()
