import torch
from torchvision import transforms
import numpy as np
from PIL import Image

class ImagePreprocessor:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def preprocess_image(self, image_path):
        """Preprocess a single image for model input"""
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        elif isinstance(image_path, Image.Image):
            image = image_path
        else:
            raise ValueError("Input must be a path string or PIL Image")
            
        # Apply transforms and return tensor
        tensor = self.transform(image)
        
        # No need to add extra dimension - DataLoader will batch for us
        return tensor

    def preprocess_from_array(self, img_array):
        """Preprocess image from numpy array"""
        # Convert numpy array to PIL Image
        img_pil = Image.fromarray(img_array)
        # Apply transformations
        img_tensor = self.transform(img_pil)
        # Add batch dimension
        return img_tensor.unsqueeze(0)

    def preprocess_from_path(self, img_path):
        """Preprocess image from file path"""
        img_pil = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img_pil)
        return img_tensor.unsqueeze(0)