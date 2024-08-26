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