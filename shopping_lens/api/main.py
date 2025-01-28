from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
import torch
from PIL import Image
import io
import os
import logging
import gc
import torch.cuda

# Import all models
from ..models.cnn_feature_extractor import CNNFeatureExtractor
from ..models.clip_model import CLIPFeatureExtractor
from ..models.vit_model import ViTFeatureExtractor
from ..models.autoencoder_model import AutoencoderFeatureExtractor
from ..utils.preprocessing import ImagePreprocessor
from ..utils.similarity import SimilaritySearch
from ..scripts.index_test_images import index_images

app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Update paths to be absolute
BASE_DIR = Path(__file__).parent.parent.absolute()
static_dir = BASE_DIR / "api" / "static"
test_images_dir = BASE_DIR / "data" / "test_images"
index_dir = BASE_DIR / "data" / "index"

# Ensure directories exist
test_images_dir.mkdir(parents=True, exist_ok=True)
index_dir.mkdir(parents=True, exist_ok=True)

# Update model paths to be absolute
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

# At the top of the file, after app initialization
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
app.mount("/images", StaticFiles(directory=str(test_images_dir)), name="images")

# Add model cache
model_cache = {}

@app.on_event("startup")
async def startup_event():
    """Initialize app on startup"""
    try:
        logger.info("Starting application initialization...")
        
        # Ensure test images exist
        if not any(test_images_dir.glob("*.jpg")):
            logger.info("Downloading test images...")
            from ..scripts.download_test_images import download_images
            download_images()
        
        logger.info("Application initialization complete")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise e

@app.get("/debug")
async def debug_info():
    """Endpoint to check system state"""
    return {
        "base_dir": str(BASE_DIR),
        "test_images": [f.name for f in test_images_dir.glob("*.jpg")],
        "indices": [f.name for f in index_dir.glob("*_index")],
        "static_files": [f.name for f in static_dir.glob("*.*")],
        "model_configs": {k: {
            "dimension": v["dimension"],
            "weights_exists": Path(v["weights_path"]).exists() if "weights_path" in v else False
        } for k, v in MODEL_CONFIGS.items()}
    }

@app.get("/")
async def root():
    return FileResponse(str(static_dir / "index.html"))

@app.get("/images/{image_name}")
async def get_image(image_name: str):
    image_path = Path(__file__).parent.parent / "data" / "test_images" / image_name
    return FileResponse(str(image_path))

def load_model(model_name, config):
    """Load model with memory management"""
    try:
        # Clear memory if needed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Check cache
        if model_name in model_cache:
            logger.info(f"Using cached model {model_name}")
            return model_cache[model_name]
        
        logger.info(f"Loading model {model_name}")
        model = config['class']()
        
        # Load weights if available
        if 'weights_path' in config:
            weights_path = Path(config['weights_path'])
            if weights_path.exists():
                model.load_state_dict(torch.load(weights_path, map_location='cpu'))
                logger.info(f"Loaded fine-tuned weights for {model_name}")
            else:
                logger.info(f"Using pretrained weights for {model_name}")
        
        # Cache the model
        model_cache[model_name] = model
        
        # Clear old models if cache is too large
        if len(model_cache) > 2:  # Keep only 2 models in memory
            oldest_model = next(iter(model_cache))
            if oldest_model != model_name:
                del model_cache[oldest_model]
                gc.collect()
                logger.info(f"Cleared {oldest_model} from cache")
        
        return model
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}")
        return None

@app.post("/search")
async def search(model_name: str, file: UploadFile):
    try:
        if model_name not in MODEL_CONFIGS:
            raise HTTPException(status_code=400, detail=f"Invalid model name: {model_name}")
        
        config = MODEL_CONFIGS[model_name]
        model = None
        
        try:
            model = load_model(model_name, config)
            if model is None:
                # Fallback to pretrained
                pretrained_name = f"{model_name}_pretrained"
                if pretrained_name in MODEL_CONFIGS:
                    logger.info(f"Falling back to pretrained model: {pretrained_name}")
                    model = load_model(pretrained_name, MODEL_CONFIGS[pretrained_name])
            
            if model is None:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to load model {model_name} and its fallback"
                )
            
            preprocessor = ImagePreprocessor()
            similarity_search = SimilaritySearch(dimension=config['dimension'])
            
            # Use the correct index suffix
            base_model_name = model_name.split('_')[0]
            index_path = BASE_DIR / "data" / "index" / f"{base_model_name}{config['index_suffix']}_index"
            
            try:
                if not index_path.exists():
                    logger.error(f"Index not found at {index_path}")
                    # Try using pretrained index if finetuned not found
                    if config['index_suffix'] == '_finetuned':
                        pretrained_index = BASE_DIR / "data" / "index" / f"{base_model_name}_pretrained_index"
                        if pretrained_index.exists():
                            logger.info(f"Using pretrained index instead: {pretrained_index}")
                            index_path = pretrained_index
                        else:
                            raise HTTPException(
                                status_code=500,
                                detail=f"No index found for {model_name}"
                            )
                    else:
                        raise HTTPException(
                            status_code=500,
                            detail=f"No index found for {model_name}"
                        )
                
                similarity_search.load_index(str(index_path))
                
            except Exception as e:
                logger.error(f"Error loading index: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to load index for {model_name}: {str(e)}"
                )
            
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert('RGB')
            
            temp_path = BASE_DIR / "data" / "temp.jpg"
            image.save(temp_path)
            
            try:
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
                logger.error(f"Error during search: {e}")
                if temp_path.exists():
                    temp_path.unlink()
                raise HTTPException(
                    status_code=500,
                    detail=f"Search failed: {str(e)}"
                )
            
        finally:
            # Clean up
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        return {"similar_images": results}
        
    except Exception as e:
        logger.error(f"Error in search: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Add health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)