import shutil
import random
from pathlib import Path
import json

def setup_data():
    current_dir = Path(__file__).parent.parent
    data_dir = current_dir / "data"
    test_dir = data_dir / "test_images"
    
    # Create directories
    for dir_name in ['train', 'val', 'test']:
        (data_dir / dir_name).mkdir(parents=True, exist_ok=True)
    
    # Get all images and their visual features
    all_images = list(test_dir.glob("*.jpg"))
    
    # Split data
    random.shuffle(all_images)
    train_size = int(0.6 * len(all_images))
    val_size = int(0.2 * len(all_images))
    
    train_images = all_images[:train_size]
    val_images = all_images[train_size:train_size+val_size]
    test_images = all_images[train_size+val_size:]
    
    # Copy images and track new paths
    new_paths = {}  # Map original paths to new relative paths
    for images, target_dir in [
        (train_images, 'train'),
        (val_images, 'val'),
        (test_images, 'test')
    ]:
        target_path = data_dir / target_dir
        for image in images:
            new_path = Path(target_dir) / image.name  # Make path relative
            shutil.copy2(image, data_dir / new_path)
            new_paths[str(image)] = str(new_path)
    
    # Create visual similarity scores with relative paths
    similarity_data = {}
    for img1 in all_images:
        base_name = img1.stem
        category1 = base_name.split('_')[0]
        new_img1_path = new_paths[str(img1)]
        similarity_data[new_img1_path] = {}
        
        for img2 in all_images:
            if img1 != img2:
                category2 = img2.stem.split('_')[0]
                new_img2_path = new_paths[str(img2)]
                # Simple similarity score based on categories
                score = 1.0 if category1 == category2 else 0.2
                similarity_data[new_img1_path][new_img2_path] = score
    
    # Save similarity scores
    with open(data_dir / 'similarity_scores.json', 'w') as f:
        json.dump(similarity_data, f, indent=4)
    
    print(f"Data split complete:")
    print(f"Train: {len(train_images)} images")
    print(f"Val: {len(val_images)} images")
    print(f"Test: {len(test_images)} images")

if __name__ == "__main__":
    setup_data() 