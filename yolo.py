import cv2
from ultralytics import YOLOv10
import supervision as sv

# Load the YOLOv10 model
model = YOLOv10('/Users/jibly/Documents/labHandling/lab_handling_system/models/best.pt')

# Open a connection to the video stream (0 is the default camera)
cap = cv2.VideoCapture('/Users/jibly/Documents/labHandling/lab_handling_system/videos/vid2.mov')

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

output_video_path = '/Users/jibly/Documents/labHandling/lab_handling_system/output/pred.mp4'

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' for .avi format
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Process the video stream frame by frame
while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame.")
        break

    # Perform inference on the frame
    # results = model(frame)

    results = model(frame, conf=0.25)[0]
    detections = sv.Detections.from_ultralytics(results)

    annotated_image = bounding_box_annotator.annotate(
        scene=frame, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections)
    
    # out.write(annotated_image)

    # sv.plot_image(annotated_image)
    cv2.imwrite('/Users/jibly/Documents/labHandling/lab_handling_system/output/stream.jpg', annotated_image)

    
    # Annotate the frame with results
    # annotated_frame = results[0].render()
    
    # # Display the frame with annotations
    # cv2.imshow('YOLOv10 Object Detection', annotated_frame)
    
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream and close windows
cap.release()
out.release()
cv2.destroyAllWindows()
