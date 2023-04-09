import os
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import glob
import gradio as gr
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry

matplotlib.pyplot.switch_backend('Agg') # for matplotlib to work in gradio
#setup model
sam_checkpoint = "sam_vit_h_4b8939.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # use GPU if available
model_type = "default"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)
predictor = SamPredictor(sam)

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))

def segment_image(image):
    masks = mask_generator.generate(image)
    plt.clf()
    ppi = 100
    height, width, _ = image.shape
    plt.figure(figsize=(width / ppi, height / ppi), dpi=ppi)
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.savefig('output.png', bbox_inches='tight', pad_inches=0)
    output = cv2.imread('output.png')
    return Image.fromarray(output)
    
with gr.Blocks() as demo:
    gr.Markdown(
        """
    # Segment Anything Model (SAM) 
    ### A test on remote sensing data
    - Paper:[(https://arxiv.org/abs/2304.02643](https://arxiv.org/abs/2304.02643)
    - Github:[https://github.com/facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)
    - Dataset:https://ai.facebook.com/datasets/segment-anything-downloads/(https://ai.facebook.com/datasets/segment-anything-downloads/)
    - Official Demo:[https://segment-anything.com/demo](https://segment-anything.com/demo)
    """
    )
    with gr.Row():
        image = gr.Image()
        image_output = gr.Image()
    print(image.shape)
    segment_image_button = gr.Button("Segment")
    segment_image_button.click(segment_image, inputs=[image], outputs=image_output)
    gr.Examples(glob.glob('./images/*'),image,image_output,segment_image)
    
demo.launch(debug=True)