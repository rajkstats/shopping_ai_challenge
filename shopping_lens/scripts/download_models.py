import torch
import timm
from transformers import CLIPProcessor, CLIPModel
from torchvision.models import ResNet50_Weights
import os
from pathlib import Path

def download_models():
    print("Downloading pretrained models...")
    cache_dir = Path("/opt/render/.cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Download ResNet50
    print("Downloading ResNet50...")
    torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    
    # Download CLIP
    print("Downloading CLIP...")
    CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    
    # Download ViT
    print("Downloading ViT...")
    timm.create_model('vit_base_patch16_224', pretrained=True)
    
    print("All models downloaded successfully!")

if __name__ == "__main__":
    download_models() 