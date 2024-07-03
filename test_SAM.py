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

# show image
plt.figure(figsize=(20,20))
plt.imshow(image)
plt.axis('off')
plt.show()

# Generate masks
masks = mask_generator.generate(image)

# print(type(masks))
print(len(masks))
print(masks[0].keys())

plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show() 