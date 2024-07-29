import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import cv2

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.inference import predict

from segment_anything import sam_model_registry, SamPredictor

def load_dino_model(config_file_path, model_path, device='cpu'):
    args = SLConfig.fromfile(config_file_path)
    dino_model = build_model(args)

    checkpoint = torch.load(model_path, map_location='cpu')
    log = dino_model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    dino_model.eval()
    dino_model = dino_model.to(device)

    return dino_model

def load_sam_model(checkpoint_path, model_type="vit_h", device="cuda"):
    sam_checkpoint = "./Tracking_SAM/pretrained_weights/sam_vit_h_4b8939.pth"  # default model
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    return predictor

def rgba2rgb(rgba, background = (255,255,255)):
    h, w, c = rgba.shape
        
    if c == 3:
        return rgba
        
    rgb = np.zeros((h, w, 3), dtype = 'float32')
    r, g, b, a = rgba[...,2], rgba[...,1], rgba[...,0], rgba[...,3]

    a = np.asanyarray(a, dtype='float32')/255

    R, G, B = background
    rgb[...,0] = r * a + (1.0 - a) * R
    rgb[...,1] = g * a + (1.0 - a) * G
    rgb[...,2] = b * a + (1.0 - a) * B

    return np.asanyarray(rgb)


class DinoGroundingSAM:
    def __init__(self):
        # 1. Load the model
        config_file_path = './Tracking_SAM/tracking_SAM/third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
        model_path = './Tracking_SAM/pretrained_weights/groundingdino_swint_ogc.pth'
        sam_checkpoint = "./Tracking_SAM/pretrained_weights/sam_vit_h_4b8939.pth" 

        self.dino_model = load_dino_model(config_file_path, model_path, device='cpu')
        self.sam_model = load_sam_model(sam_checkpoint,model_type="vit_h", device="cuda")

    def gound_dino_image_text(self, image, text_prompt='black stick', BOX_TRESHOLD = 0.3, TEXT_TRESHOLD = 0.25):
        image_np = rgba2rgb(image)
        image_np = np.asarray(image_np).astype(np.uint8)
        transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
        )
        img_chw, _ = transform(Image.fromarray(image_np), None)
        boxes, logits, phrases = predict(
            model=self.dino_model, 
            image=img_chw, 
            caption=text_prompt, 
            box_threshold=BOX_TRESHOLD, 
            text_threshold=TEXT_TRESHOLD,
            device='cpu'
        )
        H, W, _ = image_np.shape
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
        return image_np, boxes_xyxy
    
    def sam_image(self, image_np, boxes_xyxy):
        self.sam_model.set_image(image_np)
        assert len(boxes_xyxy) == 1
        input_box = boxes_xyxy[0].cpu().numpy()
        masks, _, _ = self.sam_model.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )
        return masks

