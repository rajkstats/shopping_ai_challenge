from abc import ABC, abstractmethod
import torch.nn as nn

class BaseModel(ABC, nn.Module):
    @abstractmethod
    def extract_features(self, images):
        pass