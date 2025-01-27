import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import json
from datetime import datetime

from shopping_lens.models.cnn_feature_extractor import CNNFeatureExtractor
from shopping_lens.models.clip_model import CLIPFeatureExtractor 
from shopping_lens.models.vit_model import ViTFeatureExtractor
from shopping_lens.models.autoencoder_model import AutoencoderFeatureExtractor
from shopping_lens.utils.preprocessing import ImagePreprocessor
from shopping_lens.utils.similarity import SimilaritySearch

class ShoeDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.preprocessor = ImagePreprocessor()
        self.image_paths = list(self.data_dir.glob("**/*.jpg"))
        
        # Create label mapping
        self.classes = sorted(list({p.parent.name for p in self.image_paths}))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_tensor = self.preprocessor.preprocess_image(str(img_path))
        label = self.class_to_idx[img_path.parent.name]
        return img_tensor, label, str(img_path)

def evaluate_model(model, test_loader, device):
    """Evaluate model performance on test set"""
    model.eval()
    all_features = []
    all_labels = []
    all_paths = []
    
    with torch.no_grad():
        for images, labels, paths in test_loader:
            images = images.to(device)
            features = model(images)
            
            # Handle different feature shapes
            if isinstance(features, tuple):
                features = features[0]  # Some models return (features, aux_output)
            
            # Flatten features if needed
            if features.dim() > 2:
                features = features.view(features.size(0), -1)  # Flatten to [batch_size, features]
            
            # Normalize features
            features = nn.functional.normalize(features, dim=1)
            
            all_features.append(features.cpu())
            all_labels.append(labels)
            all_paths.extend(paths)
    
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Compute similarity matrix
    similarity = torch.mm(all_features, all_features.t())
    
    # Calculate metrics
    metrics = {}
    n_samples = len(all_labels)
    
    # Adjust k values based on dataset size
    k_values = [k for k in [1, 5, 10] if k < n_samples]
    if not k_values:
        k_values = [1]  # Fallback to k=1 if dataset is very small
    
    for k in k_values:
        precision_k = []
        recall_k = []
        
        # For each query image
        for i in range(n_samples):
            # Get top k most similar images (excluding self)
            sim_scores, indices = similarity[i].topk(min(k + 1, n_samples))
            # Remove self (first element) if it's in top k
            if indices[0] == i:
                indices = indices[1:]
            else:
                indices = indices[:-1]
            
            # Handle case where k is larger than available samples
            actual_k = min(k, len(indices))
            if actual_k == 0:
                continue
                
            retrieved_labels = all_labels[indices[:actual_k]]
            
            # Calculate precision and recall
            relevant = (retrieved_labels == all_labels[i]).float()
            precision = relevant.mean().item()
            
            # Count total relevant items for this class
            total_relevant = (all_labels == all_labels[i]).sum().item() - 1  # exclude self
            if total_relevant > 0:  # Only calculate recall if there are relevant items
                recall = relevant.sum().item() / total_relevant
            else:
                recall = 0.0
            
            precision_k.append(precision)
            recall_k.append(recall)
        
        if precision_k:  # Only add metrics if we have calculations
            metrics[f'precision@{k}'] = sum(precision_k) / len(precision_k)
            metrics[f'recall@{k}'] = sum(recall_k) / len(recall_k)
    
    # Add number of test samples to metrics
    metrics['n_test_samples'] = n_samples
    
    return metrics

def train_model(model, train_loader, criterion, optimizer, device, model_type, epochs=10):
    """Train model with type-specific approach"""
    model.train()
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for images, labels, _ in progress_bar:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            if model_type == 'clip':
                # Convert labels to list for text prompts
                label_list = labels.cpu().numpy().tolist()
                # Enhanced text prompts for CLIP
                all_prompts = []
                for label_idx in label_list:
                    prompts = [
                        f"a photo of a {model.classes[label_idx]} shoe",
                        f"a {model.classes[label_idx]} shoe from a shopping website",
                        f"a professional product photo of {model.classes[label_idx]} footwear"
                    ]
                    all_prompts.extend(prompts)
                
                # Repeat each image for its multiple prompts
                images_repeated = images.repeat_interleave(3, dim=0)  # 3 prompts per image
                
                # Get image features
                image_features = model(images_repeated)
                # Get text features for all prompts
                text_features = model.encode_text(all_prompts)
                # Compute similarity
                similarity = torch.matmul(image_features, text_features.t())
                
                # Create target labels (each image matches its 3 prompts)
                target_labels = torch.arange(len(labels), device=device).repeat_interleave(3)
                loss = criterion(similarity, target_labels)
            
            elif model_type == 'autoencoder':
                # Reconstruction loss for autoencoder
                reconstructed = model(images)
                loss = criterion(reconstructed, images)
                
            elif model_type == 'vit':
                # Attention-based learning with patch masking
                features = model(images, mask_ratio=0.15)  # Mask 15% of patches
                features = nn.functional.normalize(features, dim=1)
                similarity = torch.mm(features, features.t()) / 0.07
                positive_pairs = labels.view(-1, 1) == labels.view(1, -1)
                loss = criterion(similarity, positive_pairs.float())
                
            else:  # CNN
                # Triplet loss with hard negative mining
                features = model(images)
                features = nn.functional.normalize(features, dim=1)
                
                # Get all pairwise distances
                distances = torch.cdist(features, features)
                
                # For each anchor, find hardest positive and negative
                loss = 0
                margin = 0.2
                
                for i in range(len(labels)):
                    anchor_label = labels[i]
                    pos_mask = (labels == anchor_label) & (torch.arange(len(labels), device=device) != i)
                    neg_mask = labels != anchor_label
                    
                    if pos_mask.any() and neg_mask.any():
                        # Hardest positive: most distant same-class sample
                        pos_distances = distances[i][pos_mask]
                        hardest_pos_dist = pos_distances.max()
                        
                        # Hardest negative: closest different-class sample
                        neg_distances = distances[i][neg_mask]
                        hardest_neg_dist = neg_distances.min()
                        
                        # Triplet loss with margin
                        triplet_loss = torch.clamp(hardest_pos_dist - hardest_neg_dist + margin, min=0)
                        loss += triplet_loss
                
                loss = loss / len(labels)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        avg_epoch_loss = epoch_loss / len(train_loader)
        losses.append(avg_epoch_loss)
        print(f'Epoch {epoch+1} average loss: {avg_epoch_loss:.4f}')
    
    return losses

def fine_tune_and_evaluate(train_dir, test_dir, batch_size=32, epochs=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = ShoeDataset(train_dir)
    test_dataset = ShoeDataset(test_dir)
    
    print(f"\nDataset sizes:")
    print(f"Training: {len(train_dataset)} images")
    print(f"Test: {len(test_dataset)} images")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize all models
    models = {
        'cnn': CNNFeatureExtractor(),
        'clip': CLIPFeatureExtractor(),
        'vit': ViTFeatureExtractor(),
        'autoencoder': AutoencoderFeatureExtractor()
    }
    
    results = {}
    
    # Train each model
    for name, model in models.items():
        print(f"\nTraining {name.upper()} model...")
        model = model.to(device)
        
        # Model-specific setup
        if name == 'clip':
            model.set_classes(train_dataset.classes)
        
        criterion = nn.MSELoss() if name == 'autoencoder' else nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
        # Training loop
        best_loss = float('inf')
        losses = []
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            
            for batch_idx, (images, labels, _) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                if name == 'autoencoder':
                    output = model(images)
                    loss = criterion(output, images)
                else:
                    features = model(images)
                    # Ensure features are 2D [batch_size, features]
                    if features.dim() > 2:
                        features = features.view(features.size(0), -1)
                    loss = criterion(features, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                if batch_idx % 5 == 0:
                    print(f'Epoch {epoch+1}/{epochs} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}')
            
            avg_loss = epoch_loss / len(train_loader)
            losses.append(avg_loss)
            print(f'Epoch {epoch+1} average loss: {avg_loss:.4f}')
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_path = Path(__file__).parent.parent / "models" / "weights"
                save_path.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), save_path / f"{name}_finetuned.pth")
        
        # Evaluate
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels, _ in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                if name == 'autoencoder':
                    output = model(images)
                    loss = criterion(output, images)
                else:
                    features = model(images)
                    if features.dim() > 2:
                        features = features.view(features.size(0), -1)
                    loss = criterion(features, labels)
                    pred = features.argmax(dim=1)
                    correct += pred.eq(labels).sum().item()
                
                test_loss += loss.item()
                total += labels.size(0)
        
        avg_test_loss = test_loss / len(test_loader)
        accuracy = 100. * correct / total if name != 'autoencoder' else 0
        
        results[name] = {
            'train_loss': losses[-1],
            'test_loss': avg_test_loss,
            'accuracy': accuracy
        }
        
        print(f"\n{name.upper()} Results:")
        print(f"Final Training Loss: {losses[-1]:.4f}")
        print(f"Test Loss: {avg_test_loss:.4f}")
        if name != 'autoencoder':
            print(f"Accuracy: {accuracy:.2f}%")
    
    # Save results
    results_path = Path(__file__).parent.parent / "evaluation_results" / "finetuning_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    return results

if __name__ == "__main__":
    current_dir = Path(__file__).parent.parent
    train_dir = current_dir / "data" / "train_images"
    test_dir = current_dir / "data" / "test_images"
    
    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            "Training or test data not found. "
            "Please run download_test_images.py first."
        )
    
    results = fine_tune_and_evaluate(train_dir, test_dir)
    
    # Print final summary
    print("\nFinal Results Summary:")
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}") 