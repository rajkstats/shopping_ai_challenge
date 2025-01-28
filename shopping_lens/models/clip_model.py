import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms
from .base_model import BaseModel
from pathlib import Path

class CLIPFeatureExtractor(BaseModel):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_dimension = 512
        self.classes = None  # Will be set during training
        
        # Load from cached model if available
        models_dir = Path(__file__).parent.parent / "models" / "pretrained"
        clip_dir = models_dir / "clip"
        
        if clip_dir.exists():
            print("Loading cached CLIP model...")
            self.model = CLIPModel.from_pretrained(str(clip_dir))
            self.processor = CLIPProcessor.from_pretrained(str(clip_dir))
        else:
            print("Loading pretrained CLIP model...")
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def set_classes(self, classes):
        """Set class names for text prompts"""
        self.classes = classes
    
    def encode_text(self, texts):
        """Encode text descriptions"""
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return self.model.get_text_features(**inputs)
    
    def forward(self, x):
        # Convert tensor to PIL images for CLIP
        if isinstance(x, torch.Tensor):
            if x.dim() == 4:  # batch of images
                x = [transforms.ToPILImage()(img) for img in x]
            else:  # single image
                x = [transforms.ToPILImage()(x)]
                
        inputs = self.processor(images=x, return_tensors="pt")
        # Move inputs to same device as model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        features = self.model.get_image_features(**inputs)
        return features
    
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
            
            # Get features
            features = self.forward(images)
            
            # Normalize features
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            
            return features.cpu()  # Return features on CPU for storage