import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cpu" # change to cuda if using a device with gpu

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)

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


# video_dir = "/Users/jibly/Documents/labHandling/lab_handling_system/videos"
# output_base = "/Users/jibly/Documents/labHandling/lab_handling_system/output/sam"

# videos = os.listdir(video_dir)

# count = 0

# for vid in videos:
#     if not vid.lower().endswith((".mov", ".mp4")): continue
#     vid_path = os.path.join(video_dir, vid)
#     cap = cv2.VideoCapture(vid_path)

#     while True:
#         ret, frame = cap.read()
#         if not ret: break

#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         masks = mask_generator.generate(frame)

#         fig, ax = plt.subplots(figsize=(20, 20))
#         ax.imshow(frame)
#         show_anns(masks)
#         ax.axis('off')

#         # Save the figure to the specified directory
#         output_path = os.path.join(output_base, "frame"+str(count)+".png")
#         plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
#         count+=1

#     cap.release()

# cv2.destroyAllWindows()
