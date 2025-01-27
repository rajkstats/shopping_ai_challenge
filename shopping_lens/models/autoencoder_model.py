import torch
import torch.nn as nn
from pathlib import Path
from .base_model import BaseModel

class AutoencoderFeatureExtractor(BaseModel):
    _instance = None
    _is_initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._is_initialized:
            super().__init__()
            
            # Add device attribute
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.feature_dimension = 128
            self._is_initialized = True
            
            # Encoder
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1),  # 112x112
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 56x56
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 28x28
                nn.ReLU(),
                nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 14x14
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(256 * 14 * 14, 128)
            )
            
            # Decoder
            self.decoder = nn.Sequential(
                nn.Linear(128, 256 * 14 * 14),
                nn.ReLU(),
                nn.Unflatten(1, (256, 14, 14)),
                nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
                nn.Sigmoid()
            )
            
            self.load_weights()
            # Move model to device after loading weights
            self.to(self.device)
            self.eval()
    
    def load_weights(self):
        try:
            weights_path = Path(__file__).parent / "weights" / "autoencoder.pth"
            if weights_path.exists():
                self.load_state_dict(torch.load(weights_path, map_location='cpu'))
                print("Loaded autoencoder weights successfully")
            else:
                print(f"Warning: No weights found at {weights_path}")
        except Exception as e:
            print(f"Error loading weights: {e}")
    
    def forward(self, x):
        # Don't flatten - keep original shape for convolutions
        return self.encoder(x)
    
    @torch.no_grad()
    def extract_features(self, images):
        with torch.no_grad():
            if not isinstance(images, torch.Tensor):
                raise ValueError("Input must be a tensor")
            
            # Add batch dimension if needed
            if images.dim() == 3:
                images = images.unsqueeze(0)
            
            # Ensure model is in eval mode
            self.eval()
            
            # Move images to same device as model
            images = images.to(self.device)
            
            # Get features through encoder (will handle flattening internally)
            features = self.encoder(images)
            
            # Normalize features
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            
            return features.cpu()  # Return features on CPU for storage

    def reconstruct(self, x):
        # For reconstruction, we need to encode and then decode
        encoded = self.forward(x)
        return self.decoder(encoded)