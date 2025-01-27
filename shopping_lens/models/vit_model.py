import torch
import torch.nn as nn
import timm
from .base_model import BaseModel

class ViTFeatureExtractor(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.model.head = nn.Identity()  # Remove classification head
        self.feature_dimension = 768
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            raise ValueError("Input must be a tensor")
        return self.model(x)
    
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