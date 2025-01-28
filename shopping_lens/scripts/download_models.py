import torch
import timm
from transformers import CLIPProcessor, CLIPModel
from torchvision.models import ResNet50_Weights
import os
from pathlib import Path

def download_models():
    print("Downloading pretrained models...")
    models_dir = Path(__file__).parent.parent / "models" / "pretrained"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if models are already downloaded
    if (models_dir / "resnet50.pth").exists() and \
       (models_dir / "clip").exists() and \
       (models_dir / "vit.pth").exists():
        print("Models already downloaded")
        return
    
    # Download and save ResNet50
    print("Downloading ResNet50...")
    resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    torch.save(resnet.state_dict(), models_dir / "resnet50.pth")
    
    # Download and save CLIP
    print("Downloading CLIP...")
    clip_dir = models_dir / "clip"
    clip_dir.mkdir(exist_ok=True)
    CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir=str(clip_dir))
    CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir=str(clip_dir))
    
    # Download and save ViT
    print("Downloading ViT...")
    vit = timm.create_model('vit_base_patch16_224', pretrained=True)
    torch.save(vit.state_dict(), models_dir / "vit.pth")
    
    print("All models downloaded and saved successfully!")

if __name__ == "__main__":
    download_models() 