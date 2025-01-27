from pathlib import Path
import torch
from tqdm import tqdm
from shopping_lens.models.cnn_feature_extractor import CNNFeatureExtractor
from shopping_lens.models.clip_model import CLIPFeatureExtractor
from shopping_lens.models.vit_model import ViTFeatureExtractor
from shopping_lens.models.autoencoder_model import AutoencoderFeatureExtractor
from shopping_lens.utils.preprocessing import ImagePreprocessor
from shopping_lens.utils.similarity import SimilaritySearch

def index_images(model_name, use_finetuned=False):
    suffix = "_finetuned" if use_finetuned else "_pretrained"
    print(f"\nIndexing with {model_name} model ({suffix[1:]})")
    
    # Initialize model without loading fine-tuned weights
    if model_name == 'cnn':
        model = CNNFeatureExtractor()
        dimension = 2048
    elif model_name == 'clip':
        model = CLIPFeatureExtractor()
        dimension = 512
    elif model_name == 'vit':
        model = ViTFeatureExtractor()
        dimension = 768
    else:  # autoencoder
        model = AutoencoderFeatureExtractor()
        dimension = 128
    
    # Load appropriate weights
    if use_finetuned:
        weights_path = Path(__file__).parent.parent / "models" / "weights" / f"{model_name}_finetuned.pth"
        if weights_path.exists():
            model.load_state_dict(torch.load(weights_path, map_location='cpu'))
            print(f"Loaded fine-tuned weights for {model_name}")
        else:
            print(f"Warning: No fine-tuned weights found for {model_name}")
            return
    elif model_name == 'autoencoder':
        # Only load pre-trained weights for autoencoder
        weights_path = Path(__file__).parent.parent / "models" / "weights" / "autoencoder.pth"
        if weights_path.exists():
            model.load_state_dict(torch.load(weights_path, map_location='cpu'))
            print("Loaded pre-trained autoencoder weights")
    
    # Move model to appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Initialize preprocessor and similarity search
    preprocessor = ImagePreprocessor()
    similarity_search = SimilaritySearch(dimension=dimension)
    
    # Get images from directory
    current_dir = Path(__file__).parent.parent
    image_dir = current_dir / "data" / "test_images"
    
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
            
    image_files = list(image_dir.glob("*.jpg"))
    print(f"Found {len(image_files)} images to index")
    
    # Process images one by one
    for image_path in tqdm(image_files):
        image = preprocessor.preprocess_image(str(image_path))
        features = model.extract_features(image)
        # Squeeze out batch dimension and convert to numpy
        features = features.squeeze(0).numpy()
        similarity_search.add_item(features, str(image_path))
        
    # Save index with appropriate suffix
    index_dir = current_dir / "data" / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = str(index_dir / f"{model_name}{suffix}_index")
    similarity_search.save_index(index_path)
    
    print(f"\nSuccessfully indexed {len(image_files)} images for {model_name} ({suffix[1:]})")

if __name__ == "__main__":
    # First remove existing indices
    import shutil
    index_dir = Path(__file__).parent.parent / "data" / "index"
    if index_dir.exists():
        shutil.rmtree(index_dir)
        print("Removed existing indices")
    
    # Index all models (both pre-trained and fine-tuned)
    for model_name in ['cnn', 'clip', 'vit', 'autoencoder']:
        # Index pre-trained version
        try:
            index_images(model_name, use_finetuned=False)
        except Exception as e:
            print(f"Error indexing pre-trained {model_name}: {e}")
        
        # Index fine-tuned version
        try:
            index_images(model_name, use_finetuned=True)
        except Exception as e:
            print(f"Error indexing fine-tuned {model_name}: {e}")