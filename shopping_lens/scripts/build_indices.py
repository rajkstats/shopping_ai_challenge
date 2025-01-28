from pathlib import Path
from index_test_images import index_images

def build_all_indices():
    """Build all indices and save them to the repository"""
    for model_name in ['cnn', 'clip', 'vit', 'autoencoder']:
        print(f"Building indices for {model_name}...")
        index_images(model_name, use_finetuned=False)
        index_images(model_name, use_finetuned=True)

if __name__ == "__main__":
    build_all_indices() 