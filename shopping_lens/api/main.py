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
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.cors import CORSMiddleware

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
        'weights_path': str(BASE_DIR / 'models/weights/clip_finetuned.pth'),
        'pretrained_path': str(BASE_DIR / 'models/pretrained/clip_model'),
        'processor_path': str(BASE_DIR / 'models/pretrained/clip_processor'),
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

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add gzip compression
app.add_middleware(GZipMiddleware, minimum_size=500)

# Mount static files
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
app.mount("/images", StaticFiles(directory=str(test_images_dir)), name="images")

# Add cache control middleware
@app.middleware("http")
async def add_cache_control_header(request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/images/"):
        response.headers["Cache-Control"] = "public, max-age=3600"
    return response

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
        if model_name in model_cache:
            logger.info(f"Using cached model {model_name}")
            return model_cache[model_name]
        
        logger.info(f"Loading model {model_name}")
        model = config['class']()
        
        # Load pretrained weights first if available
        if 'pretrained_path' in config:
            pretrained_path = Path(config['pretrained_path'])
            if pretrained_path.exists():
                if isinstance(model, CLIPFeatureExtractor):
                    model.load_pretrained(
                        config['pretrained_path'],
                        config['processor_path']
                    )
                else:
                    model.load_state_dict(torch.load(pretrained_path, map_location='cpu'))
                logger.info(f"Loaded pretrained weights for {model_name}")
        
        # Then load fine-tuned weights if available
        if 'weights_path' in config:
            weights_path = Path(config['weights_path'])
            if weights_path.exists():
                model.load_state_dict(torch.load(weights_path, map_location='cpu'))
                logger.info(f"Loaded fine-tuned weights for {model_name}")
        
        model_cache[model_name] = model
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
        logger.info(f"Processing request for model: {model_name}")
        
        # Check index first
        base_model_name = model_name.split('_')[0]
        index_path = BASE_DIR / "data" / "index" / f"{base_model_name}{config['index_suffix']}_index"
        faiss_path = Path(str(index_path) + ".faiss")
        meta_path = Path(str(index_path) + ".meta")
        
        # Debug logging
        logger.info(f"Looking for index files at:")
        logger.info(f"FAISS path: {faiss_path} (exists: {faiss_path.exists()})")
        logger.info(f"Meta path: {meta_path} (exists: {meta_path.exists()})")
        logger.info(f"Base directory: {BASE_DIR}")
        logger.info(f"Available indices: {list(index_dir.glob('*'))}")
        
        if not (faiss_path.exists() and meta_path.exists()):
            raise HTTPException(
                status_code=500,
                detail=f"Index not found for {model_name}. Available indices: {list(index_dir.glob('*'))}"
            )
        
        # Then load model
        model = load_model(model_name, config)
        if model is None:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load model {model_name}"
            )
            
        preprocessor = ImagePreprocessor()
        similarity_search = SimilaritySearch(dimension=config['dimension'])
        
        # Use the correct index suffix
        base_model_name = model_name.split('_')[0]
        index_path = BASE_DIR / "data" / "index" / f"{base_model_name}{config['index_suffix']}_index"
        faiss_path = Path(str(index_path) + ".faiss")
        meta_path = Path(str(index_path) + ".meta")
        
        if not (faiss_path.exists() and meta_path.exists()):
            logger.error(f"Index files not found at {index_path}")
            logger.error(f"FAISS exists: {faiss_path.exists()}, Meta exists: {meta_path.exists()}")
            # Try using pretrained index if finetuned not found
            if config['index_suffix'] == '_finetuned':
                pretrained_base = BASE_DIR / "data" / "index" / f"{base_model_name}_pretrained_index"
                pretrained_faiss = Path(str(pretrained_base) + ".faiss")
                pretrained_meta = Path(str(pretrained_base) + ".meta")
                if pretrained_faiss.exists() and pretrained_meta.exists():
                    logger.info(f"Using pretrained index instead: {pretrained_base}")
                    index_path = pretrained_base
                    faiss_path = pretrained_faiss
                    meta_path = pretrained_meta
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
        
    except Exception as e:
        logger.error(f"Error in search: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Add health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/check-indices")
async def check_indices():
    """Debug endpoint to check index files"""
    try:
        indices = {}
        for model_name in ['cnn', 'clip', 'vit', 'autoencoder']:
            for suffix in ['_pretrained', '_finetuned']:
                index_path = BASE_DIR / "data" / "index" / f"{model_name}{suffix}_index"
                faiss_path = Path(str(index_path) + ".faiss")
                meta_path = Path(str(index_path) + ".meta")
                indices[f"{model_name}{suffix}"] = {
                    "index_exists": index_path.exists(),
                    "faiss_exists": faiss_path.exists(),
                    "meta_exists": meta_path.exists(),
                    "index_path": str(index_path),
                    "faiss_path": str(faiss_path),
                    "meta_path": str(meta_path)
                }
        return {
            "base_dir": str(BASE_DIR),
            "index_dir": str(index_dir),
            "indices": indices,
            "files_in_index_dir": [str(f) for f in index_dir.glob("*")]
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)