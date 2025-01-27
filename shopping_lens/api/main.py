from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
import torch
from PIL import Image
import io
import os

# Import all models
from ..models.cnn_feature_extractor import CNNFeatureExtractor
from ..models.clip_model import CLIPFeatureExtractor
from ..models.vit_model import ViTFeatureExtractor
from ..models.autoencoder_model import AutoencoderFeatureExtractor
from ..utils.preprocessing import ImagePreprocessor
from ..utils.similarity import SimilaritySearch
from ..scripts.index_test_images import index_images

app = FastAPI()

# Update the static files directory path
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Update model paths to be relative to the project root
BASE_DIR = Path(__file__).parent.parent

MODEL_CONFIGS = {
    'cnn': {
        'class': CNNFeatureExtractor,
        'dimension': 2048,
        'weights_path': str(BASE_DIR / 'models/weights/cnn_finetuned.pth'),
        'index_suffix': '_finetuned'
    },
    'cnn_pretrained': {
        'class': CNNFeatureExtractor,
        'dimension': 2048,
        'index_suffix': '_pretrained'
    },
    'clip': {
        'class': CLIPFeatureExtractor,
        'dimension': 512,
        'weights_path': 'models/weights/clip_finetuned.pth',
        'index_suffix': '_finetuned'
    },
    'clip_pretrained': {
        'class': CLIPFeatureExtractor,
        'dimension': 512,
        'index_suffix': '_pretrained'
    },
    'vit': {
        'class': ViTFeatureExtractor,
        'dimension': 768,
        'weights_path': 'models/weights/vit_finetuned.pth',
        'index_suffix': '_finetuned'
    },
    'vit_pretrained': {
        'class': ViTFeatureExtractor,
        'dimension': 768,
        'index_suffix': '_pretrained'
    },
    'autoencoder': {
        'class': AutoencoderFeatureExtractor,
        'dimension': 128,
        'weights_path': 'models/weights/autoencoder_finetuned.pth',
        'index_suffix': '_finetuned'
    },
    'autoencoder_pretrained': {
        'class': AutoencoderFeatureExtractor,
        'dimension': 128,
        'weights_path': 'models/weights/autoencoder.pth',  # Pre-trained autoencoder has its own weights
        'index_suffix': '_pretrained'
    }
}

# Add startup event
@app.on_event("startup")
async def startup_event():
    """Create indices on startup if they don't exist"""
    try:
        # Check if indices exist
        index_dir = Path(__file__).parent.parent / "data" / "index"
        if not index_dir.exists() or not any(index_dir.iterdir()):
            print("Creating indices...")
            # Create indices for all models
            for model_name in ['cnn', 'clip', 'vit', 'autoencoder']:
                try:
                    index_images(model_name, use_finetuned=False)
                    index_images(model_name, use_finetuned=True)
                except Exception as e:
                    print(f"Error indexing {model_name}: {e}")
            print("Indices created successfully")
    except Exception as e:
        print(f"Error during startup: {e}")

@app.get("/")
async def root():
    return FileResponse(str(static_dir / "index.html"))

@app.get("/images/{image_name}")
async def get_image(image_name: str):
    image_path = Path(__file__).parent.parent / "data" / "test_images" / image_name
    return FileResponse(str(image_path))

@app.post("/search")
async def search(model_name: str, file: UploadFile):
    try:
        if model_name not in MODEL_CONFIGS:
            raise HTTPException(status_code=400, detail=f"Invalid model name: {model_name}")
            
        config = MODEL_CONFIGS[model_name]
        model = config['class']()
        
        # Load weights if available
        if 'weights_path' in config:
            weights_path = Path(config['weights_path'])
            if weights_path.exists():
                model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        
        preprocessor = ImagePreprocessor()
        similarity_search = SimilaritySearch(dimension=config['dimension'])
        
        # Use the correct index suffix
        base_model_name = model_name.split('_')[0]
        index_path = BASE_DIR / "data" / "index" / f"{base_model_name}{config['index_suffix']}_index"
        similarity_search.load_index(str(index_path))
        
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        temp_path = BASE_DIR / "data" / "temp.jpg"
        image.save(temp_path)
        
        image_tensor = preprocessor.preprocess_image(str(temp_path))
        features = model.extract_features(image_tensor)
        paths, scores = similarity_search.search(features, k=5)
        
        if temp_path.exists():
            temp_path.unlink()
        
        results = [
            {"path": Path(path).name, "similarity": float(score)} 
            for path, score in zip(paths, scores)
        ]
        
        return {"similar_images": results}
        
    except Exception as e:
        print(f"Error in search: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)