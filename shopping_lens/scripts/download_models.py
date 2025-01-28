import torch
import timm
from transformers import CLIPProcessor, CLIPModel
from torchvision.models import ResNet50_Weights
import os
from pathlib import Path

def download_models():
    print("Downloading pretrained models...")
    cache_dir = Path("/opt/render/.cache")
    models_dir = Path(__file__).parent.parent / "models" / "pretrained"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Download and save ResNet50
    print("Downloading ResNet50...")
    resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    torch.save(resnet.state_dict(), models_dir / "resnet50_pretrained.pth")
    
    # Download and save CLIP
    print("Downloading CLIP...")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor.save_pretrained(models_dir / "clip_processor")
    clip_model.save_pretrained(models_dir / "clip_model")
    
    # Download and save ViT
    print("Downloading ViT...")
    vit = timm.create_model('vit_base_patch16_224', pretrained=True)
    torch.save(vit.state_dict(), models_dir / "vit_pretrained.pth")
    
    print("All models downloaded and saved successfully!")

if __name__ == "__main__":
    download_models() 