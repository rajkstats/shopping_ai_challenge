import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
from .base_model import BaseModel

class CNNFeatureExtractor(BaseModel):
    def __init__(self):
        super().__init__()
        # Initialize ResNet50 without classification head
        self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # Remove the final FC layer
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.feature_dimension = 2048
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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