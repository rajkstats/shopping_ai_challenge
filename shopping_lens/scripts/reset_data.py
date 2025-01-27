import shutil
from pathlib import Path

def reset_data():
    current_dir = Path(__file__).parent.parent
    
    # Delete test images
    test_images_dir = current_dir / "data" / "test_images"
    if test_images_dir.exists():
        shutil.rmtree(test_images_dir)
        print(f"Deleted test images directory: {test_images_dir}")
    
    # Delete index files
    index_dir = current_dir / "data" / "index"
    if index_dir.exists():
        shutil.rmtree(index_dir)
        print(f"Deleted index directory: {index_dir}")
        
    # Delete any FAISS index files in data directory
    data_dir = current_dir / "data"
    if data_dir.exists():
        for file in data_dir.glob("*index*"):
            file.unlink()
            print(f"Deleted index file: {file}")

if __name__ == "__main__":
    reset_data()