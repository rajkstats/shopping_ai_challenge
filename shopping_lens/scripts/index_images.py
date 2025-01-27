import torch
from pathlib import Path
from tqdm import tqdm
import argparse

from shopping_lens.models.cnn_feature_extractor import CNNFeatureExtractor
from shopping_lens.models.clip_model import CLIPFeatureExtractor
from shopping_lens.models.vit_model import ViTFeatureExtractor
from shopping_lens.models.autoencoder_model import AutoencoderFeatureExtractor
from shopping_lens.utils.similarity import SimilaritySearch
from shopping_lens.utils.preprocessing import ImagePreprocessor

def index_images(image_dir, output_path, model_name="clip-ViT-B-32"):
    # Initialize components
    model = FeatureExtractor(model_name)
    preprocessor = ImagePreprocessor()
    similarity_search = SimilaritySearch(model.feature_dimension)
    
    # Get all image files
    image_dir = Path(image_dir)
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
        image_files.extend(image_dir.rglob(ext))
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    for image_path in tqdm(image_files):
        image = preprocessor.preprocess_image(str(image_path))
        features = model.extract_features(image)
        similarity_search.add_item(features, str(image_path))
    
    # Save index
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    similarity_search.save_index(str(output_path))
    
    print(f"Index saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Index images for similarity search')
    parser.add_argument('image_dir', help='Directory containing images to index')
    parser.add_argument('output_path', help='Path to save the index')
    parser.add_argument('--model', default='clip-ViT-B-32', help='Model to use for feature extraction')
    
    args = parser.parse_args()
    
    index_images(args.image_dir, args.output_path, args.model)

if __name__ == "__main__":
    main()