from PIL import Image
import torch
import cv2
import sys
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

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
    ax.imshow(img)


sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
#device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
#sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)

# # Load your image and convert it to a NumPy array
# image_path = "/Users/y.i_2.0/Documents/course/lab_handling_system/test.jpg"
# image = Image.open(image_path)
# image_np = np.array(image)  # Convert the image to a NumPy array
image = cv2.imread('/Users/y.i_2.0/Documents/course/lab_handling_system/test.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Generate masks
masks = mask_generator.generate(image)

# print(type(masks))
print(len(masks))
print(masks[0].keys())

# plt.figure(figsize=(20,20))
# plt.imshow(image)
# show_anns(masks)
# plt.axis('off')
# plt.show() 

masked_image = image.copy()

# Function to display annotations
def show_anns(masks, image):
    if len(masks) == 0:
        return
    sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))

# Plot and overlay masks
plt.figure(figsize=(20, 20))
plt.imshow(image)
show_anns(masks, image)
plt.axis('off')
plt.show()

plt.savefig('/Users/y.i_2.0/Documents/course/lab_handling_system/SAM_output2.jpg', bbox_inches='tight', pad_inches=0)

# Initialize the YOLO model
model = YOLOv10('/Users/y.i_2.0/Documents/course/lab_handling_system/models/best.pt')
image_path = '/Users/y.i_2.0/Documents/course/lab_handling_system/SAM_output.jpg'
image = cv2.imread(image_path)

if image is None:
    print("Error: Could not load image.")
    exit()

# Setup annotation tools
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

def annotate_image(image):
    results = model(image, conf=0.3)[0]
    detections = sv.Detections.from_ultralytics(results)

    if not detections:
        print("No detections")
        return image

    # Annotate the image with bounding boxes and labels
    annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
    return annotated_image

# Process the image
annotated_image = annotate_image(image)

# Save the output
output_path = '/Users/y.i_2.0/Documents/course/lab_handling_system/annotated_image.jpg'
cv2.imwrite(output_path, annotated_image)
print(f"Annotated image saved to {output_path}")

# Optionally display the image
plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
plt.show()
