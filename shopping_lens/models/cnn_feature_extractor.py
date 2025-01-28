import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
from .base_model import BaseModel
from pathlib import Path

class CNNFeatureExtractor(BaseModel):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_dimension = 2048
        
        # Load from cached model if available
        models_dir = Path(__file__).parent.parent / "models" / "pretrained"
        cached_path = models_dir / "resnet50.pth"
        
        if cached_path.exists():
            print("Loading cached ResNet50 model...")
            self.model = models.resnet50()
            self.model.load_state_dict(torch.load(cached_path, map_location=self.device))
        else:
            print("Loading pretrained ResNet50 model...")
            self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        
        # Remove the final FC layer
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model = self.model.to(self.device)
        self.model.eval()

    def forward(self, x):
        # Forward pass through the network
        x = self.model(x)
        # Flatten the output
        return torch.flatten(x, 1)

    @torch.no_grad()
    def extract_features(self, images):
        with torch.no_grad():
            if not isinstance(images, torch.Tensor):
                raise ValueError("Input must be a tensor")
            
            # Add batch dimension if needed
            if images.dim() == 3:
                images = images.unsqueeze(0)
            
            # Ensure model is in eval mode
            self.model.eval()
            
            # Move images to same device as model
            images = images.to(self.device)
            
            # Get features
            features = self.forward(images)
            
            # Normalize features
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            
            return features.cpu()  # Return features on CPU for storage

    def load_state_dict(self, state_dict):
        # Only load state dict if we want to override pretrained weights
        self.model.load_state_dict(state_dict)
        self.model.eval()