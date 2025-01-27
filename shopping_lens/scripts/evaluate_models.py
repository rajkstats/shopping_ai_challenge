import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import json

from shopping_lens.utils.similarity import SimilaritySearch
from shopping_lens.utils.preprocessing import ImagePreprocessor
from shopping_lens.models.clip_model import CLIPFeatureExtractor
from shopping_lens.models.vit_model import ViTFeatureExtractor
from shopping_lens.models.cnn_feature_extractor import CNNFeatureExtractor
from shopping_lens.models.autoencoder_model import AutoencoderFeatureExtractor

def evaluate_model(model_name, test_data_path, batch_size=32):
    # Initialize model based on type
    if model_name == 'clip':  # Changed from clip-ViT-B-32 to just 'clip'
        model = CLIPFeatureExtractor()  # Remove model_name parameter if not needed
    elif model_name == 'cnn':
        model = CNNFeatureExtractor()
    elif model_name == 'vit':
        model = ViTFeatureExtractor()
    elif model_name == 'autoencoder':
        model = AutoencoderFeatureExtractor()
    else:
        raise ValueError(f"Unsupported model: {model_name}")
        
    preprocessor = ImagePreprocessor()
    similarity_search = SimilaritySearch(model.feature_dimension)
    
    
    # Load test data
    test_data = Path(test_data_path)
    query_folders = [f for f in test_data.iterdir() if f.is_dir()]
    
    results = {}
    
    for query_folder in tqdm(query_folders, desc="Processing query folders"):
        # Get query image and similar images
        query_image = next(query_folder.glob("query.*"))
        similar_images = list(query_folder.glob("similar*"))
        
        # Process all images in folder
        all_images = [query_image] + similar_images
        all_paths = [str(img) for img in all_images]
        all_features = []
        
        # Extract features for all images
        for image_path in all_paths:
            image = preprocessor.preprocess_image(image_path)
            features = model.extract_features(image)
            all_features.append(features.cpu().numpy())
            
        # Add features to index
        features_array = np.vstack(all_features).astype(np.float32)
        for feat, path in zip(features_array, all_paths):
            similarity_search.add_item(feat, path)
            
        # Search using query image features
        query_features = all_features[0]
        retrieved_paths, similarity_scores = similarity_search.search(query_features, k=len(all_images))
        
        # Calculate metrics
        relevant_paths = set(str(img) for img in similar_images)
        retrieved_relevant = [path for path in retrieved_paths if path in relevant_paths]
        
        precision = len(retrieved_relevant) / len(retrieved_paths) if retrieved_paths else 0
        recall = len(retrieved_relevant) / len(relevant_paths) if relevant_paths else 0
        
        results[query_folder.name] = {
            "precision": precision,
            "recall": recall,
            "retrieved_paths": retrieved_paths,
            "similarity_scores": similarity_scores
        }
        
        # Clear index for next query
        similarity_search = SimilaritySearch(model.feature_dimension)
    
    # Calculate average metrics
    avg_precision = np.mean([r["precision"] for r in results.values()])
    avg_recall = np.mean([r["recall"] for r in results.values()])
    
    return {
        "model_name": model_name,
        "average_precision": float(avg_precision),
        "average_recall": float(avg_recall),
        "detailed_results": results
    }

def main():
    # Model configurations to evaluate
    models = [
        "clip-ViT-B-32",
        "clip-ViT-L-14",
        "openai-clip-vit-large-patch14",
    ]
    
    test_data_path = "data/test_queries"
    results = {}
    
    for model_name in models:
        print(f"\nEvaluating {model_name}...")
        model_results = evaluate_model(model_name, test_data_path)
        results[model_name] = model_results
        
        print(f"Average Precision: {model_results['average_precision']:.4f}")
        print(f"Average Recall: {model_results['average_recall']:.4f}")
    
    # Save results
    output_path = Path("evaluation_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()