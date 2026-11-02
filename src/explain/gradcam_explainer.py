"""Grad-CAM heatmaps showing which regions push a frame's embedding away from 'normal'."""
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

class _EmbeddingModel(nn.Module):
    """Wraps the ResNet body so the output is the flat 512-d embedding."""
    def __init__(self, body):
        super(_EmbeddingModel, self).__init__()
        super().__init__()
        self.body = body

    def forward(self, x):
        return self.body(x).flatten(1)
    
class _DirectionTarget:
    """Grad-CAM target: how strongly the embedding moves along 'direction'."""
    def __init__(self, direction):
        self.direction = direction
        
    def __call__(self, model_output):
        return (model_output * self.direction).sum()
    
class GradCAMExplainer:
    def __init__(self, extractor):
        self.extractor = extractor
        self.model = _EmbeddingModel(extractor.body).eval()
        self.target_layer = extractor.backbone.layer4[-1]  #last conv block, 7x7 map
        
    def explain(self, img: Image.Image, direction: np.ndarray):
        """direction = current embedding - window mean.
        Returns (cam in [0,1] of shape 224x224, RGB overlay uint8)."""
        #weights are frozen, so the input must require grad for hooks to capture gradients
        x = self.extractor.preprocess(img).requires_grad_(True)
        d = torch.from_numpy(direction).float().to(self.extractor.device)
        d = d / (d.norm() + 1e-8)
        
        with GradCAM(model=self.model, target_layers=[self.target_layer]) as cam:
            gray = cam(input_tensor=x, targets=[_DirectionTarget(d)])[0]
            
        rgb = np.asarray(img.convert("RGB").resize((224, 224)), dtype=np.float32) / 255.0
        overlay = show_cam_on_image(rgb, gray, use_rgb=True)
        return gray, overlay