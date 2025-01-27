import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import shutil
from shopping_lens.models.autoencoder_model import AutoencoderFeatureExtractor

def setup_training_data():
    # Setup paths
    base_dir = Path(__file__).parent.parent
    train_dir = base_dir / "data" / "train_images" / "shoes"  # Added 'shoes' subdirectory for ImageFolder
    test_images_dir = base_dir / "data" / "test_images"
    
    # Create training directory
    train_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy test images to training directory if it's empty
    if not any(train_dir.iterdir()):
        for img_path in test_images_dir.glob("*.jpg"):
            shutil.copy2(img_path, train_dir)
        print(f"Copied training images to {train_dir}")
    
    return train_dir

def train_autoencoder():
    # Setup training data
    train_dir = setup_training_data()
    
    # Initialize model
    model = AutoencoderFeatureExtractor()
    
    # Define training parameters
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Load your training data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    train_dataset = datasets.ImageFolder(train_dir.parent, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Training loop
    num_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            # Forward pass - reconstruct the input
            reconstructed = model(data)
            
            # Compute reconstruction loss
            loss = criterion(reconstructed, data)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')
    
    # Save the trained model
    weights_dir = Path(__file__).parent.parent / "models" / "weights"
    weights_dir.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), weights_dir / 'autoencoder.pth')

if __name__ == "__main__":
    train_autoencoder()