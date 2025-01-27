import faiss
import numpy as np
import torch
import pickle
from pathlib import Path
import imagehash
from PIL import Image
import time

class SimilaritySearch:
    def __init__(self, dimension):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.paths = []
        self.image_hashes = {}
    
    def add_item(self, features, image_path):
        """Add a single item to the index with duplicate checking"""
        # Generate image hash
        img_hash = self._compute_image_hash(image_path)
        
        # Check if this image is a duplicate
        for existing_hash in self.image_hashes.values():
            if self._is_similar_hash(img_hash, existing_hash):
                return  # Skip duplicate image
                
        # Add new image
        self.image_hashes[image_path] = img_hash
        self.paths.append(image_path)
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
        features = features.reshape(1, -1).astype('float32')
        self.index.add(features)
        
    def _compute_image_hash(self, image_path):
        """Compute perceptual hash of image"""
        with Image.open(image_path) as img:
            return str(imagehash.average_hash(img))
            
    def _is_similar_hash(self, hash1, hash2, threshold=5):
        """Check if two image hashes are similar"""
        # Convert hex strings to binary strings
        bin1 = bin(int(hash1, 16))[2:].zfill(64)
        bin2 = bin(int(hash2, 16))[2:].zfill(64)
        # Calculate Hamming distance
        return sum(c1 != c2 for c1, c2 in zip(bin1, bin2)) <= threshold

    def search(self, query_features, k=5):
        if isinstance(query_features, torch.Tensor):
            query_features = query_features.cpu().numpy()
        query_features = query_features.reshape(1, -1).astype('float32')
        
        # Get distances and indices
        distances, indices = self.index.search(query_features, k)
        
        # Convert L2 distances to similarity scores (0-100)
        max_distance = np.max(distances) + 1e-6  # Avoid division by zero
        similarities = 100 * (1 - distances / max_distance)
        
        # Get paths and scores
        results = []
        seen_hashes = set()
        
        for idx, similarity in zip(indices[0], similarities[0]):
            if idx < len(self.paths):
                path = self.paths[idx]
                img_hash = self.image_hashes[path]
                
                # Check for duplicates
                is_duplicate = any(self._is_similar_hash(img_hash, h) for h in seen_hashes)
                if not is_duplicate:
                    results.append({
                        'path': path,
                        'similarity': float(similarity)
                    })
                    seen_hashes.add(img_hash)
        
        # Sort by similarity in descending order
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Return paths and scores separately
        paths = [r['path'] for r in results]
        scores = [r['similarity'] for r in results]
        
        return paths, scores

    def save_index(self, path):
        # Save FAISS index
        faiss.write_index(self.index, f"{path}.faiss")
        # Save paths and hashes
        data = {
            'paths': self.paths,
            'hashes': self.image_hashes
        }
        with open(f"{path}.meta", 'wb') as f:
            pickle.dump(data, f)

    def load_index(self, path):
        # Load FAISS index
        self.index = faiss.read_index(f"{path}.faiss")
        # Load paths and hashes
        with open(f"{path}.meta", 'rb') as f:
            data = pickle.load(f)
            self.paths = data['paths']
            self.image_hashes = data['hashes']

def evaluate_similarity_search(model, loader, device, k=5):
    """Evaluate similarity search performance"""
    model.eval()
    
    # Metrics
    retrieval_time = []
    similarity_accuracy = []
    
    with torch.no_grad():
        for batch in loader:
            start_time = time.time()
            
            # Extract features
            query_features = model(batch['image1'].to(device))
            
            # Search similar items
            similarities = compute_similarities(query_features, database_features)
            top_k = get_top_k(similarities, k=k)
            
            # Measure time
            retrieval_time.append(time.time() - start_time)
            
            # Compare with ground truth
            target_sim = batch['similarity']
            accuracy = compute_ranking_correlation(similarities, target_sim)
            similarity_accuracy.append(accuracy)
    
    return {
        'avg_retrieval_time': np.mean(retrieval_time),
        'similarity_accuracy': np.mean(similarity_accuracy),
        'memory_usage': get_model_memory_usage(model)
    }