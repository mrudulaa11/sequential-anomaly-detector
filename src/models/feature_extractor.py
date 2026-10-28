"""Frozen ResNet18 embedding extractor (512-d per frame)."""
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

class FeatureExtractor:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad_(False)
        self.backbone = backbone.to(self.device) #full-model, reused by Grad-CAM later
        #Everything except the final fc layer -> output (1, 512, 1, 1)
        self.body = nn.Sequential(*list(self.backbone.children())[:-1])
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        
    def preprocess(self, img: Image.Image) -> torch.Tensor:
        return self.transform(img.convert("RGB")).unsqueeze(0).to(self.device)
    
    @torch.no_grad()
    def embed(self, img: Image.Image) -> np.ndarray:
        feats = self.body(self.preprocess(img))
        return feats.flatten(1).squeeze(0).cpu().numpy()
    