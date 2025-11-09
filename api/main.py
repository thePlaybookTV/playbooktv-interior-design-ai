"""
PlaybookTV Interior Design AI - FastAPI Server
Production-ready API for Modomo app integration
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import torch
import numpy as np
from PIL import Image
import io
import logging
from pathlib import Path
import os
from datetime import datetime

# Setup logging FIRST (before anything else)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fix Python path to find src module
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import your models
try:
    from src.models.improved_style_classifier import EfficientNetStyleClassifier, EnsembleStyleClassifier
    from ultralytics import YOLO
    logger.info("✅ Model imports successful")
except ImportError as e:
    logger.warning(f"⚠️ Could not import models: {e}")
    logger.warning("Models will be loaded dynamically when available")
    EfficientNetStyleClassifier = None
    EnsembleStyleClassifier = None
    YOLO = None

# Initialize FastAPI
app = FastAPI(
    title="PlaybookTV Interior Design AI",
    description="AI-powered interior design analysis API",
    version="2.0.0"
)

# CORS settings for Modomo app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model storage
models = {
    "yolo": None,
    "style_ensemble": None,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# Configuration
class Config:
    # Model paths (configure these)
    YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "./models/yolo_best.pt")
    EFFICIENTNET_PATH = os.getenv("EFFICIENTNET_PATH", "./models/best_efficientnet_style_classifier.pth")
    RESNET_PATH = os.getenv("RESNET_PATH", "./models/best_resnet_style_classifier.pth")
    VIT_PATH = os.getenv("VIT_PATH", "./models/best_vit_style_classifier.pth")

    # Categories
    ROOM_TYPES = ['living_room', 'bedroom', 'kitchen', 'dining_room', 'bathroom', 'home_office']
    STYLES = ['modern', 'traditional', 'contemporary', 'minimalist', 'scandinavian',
              'industrial', 'bohemian', 'mid_century_modern', 'rustic']

config = Config()

# Response models
class DetectionResult(BaseModel):
    item_type: str
    confidence: float
    bbox: List[float]
    area_percentage: float

class StylePrediction(BaseModel):
    style: str
    confidence: float
    all_probabilities: Dict[str, float]

class RoomPrediction(BaseModel):
    room_type: str
    confidence: float

class AnalysisResult(BaseModel):
    detections: List[DetectionResult]
    detection_count: int
    style: StylePrediction
    room: Optional[RoomPrediction] = None
    processing_time_ms: float

# ============================================
# MODEL LOADING
# ============================================

@app.on_event("startup")
async def load_models():
    """Load all models on startup"""
    logger.info("Loading models...")

    try:
        # Load YOLO
        if Path(config.YOLO_MODEL_PATH).exists():
            logger.info(f"Loading YOLO from {config.YOLO_MODEL_PATH}")
            models["yolo"] = YOLO(config.YOLO_MODEL_PATH)
            logger.info("✅ YOLO loaded")
        else:
            logger.warning(f"⚠️ YOLO model not found at {config.YOLO_MODEL_PATH}")

        # Load Style Ensemble
        if all(Path(p).exists() for p in [config.EFFICIENTNET_PATH, config.RESNET_PATH, config.VIT_PATH]):
            logger.info("Loading Style Ensemble...")
            ensemble = EnsembleStyleClassifier(
                num_styles=len(config.STYLES),
                device=models["device"]
            )

            # Load weights
            ensemble.models['efficientnet'].load_state_dict(
                torch.load(config.EFFICIENTNET_PATH, map_location=models["device"])
            )
            ensemble.models['resnet'].load_state_dict(
                torch.load(config.RESNET_PATH, map_location=models["device"])
            )
            ensemble.models['vit'].load_state_dict(
                torch.load(config.VIT_PATH, map_location=models["device"])
            )

            # Set to eval mode
            for model in ensemble.models.values():
                model.eval()

            models["style_ensemble"] = ensemble
            logger.info("✅ Style Ensemble loaded")
        else:
            logger.warning("⚠️ Style classifier models not found")

        logger.info(f"🎮 Using device: {models['device']}")
        logger.info("✅ All models loaded successfully")

    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        raise

# ============================================
# HELPER FUNCTIONS
# ============================================

def preprocess_image(image: Image.Image, size: tuple = (224, 224)):
    """Preprocess image for model inference"""
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return transform(image).unsqueeze(0)

def extract_furniture_context(detections: List[dict]) -> torch.Tensor:
    """Extract context features from detections"""
    if not detections:
        return torch.zeros(3, dtype=torch.float32)

    furniture_count = len(detections)
    avg_area = np.mean([d['area_percentage'] for d in detections])
    avg_confidence = np.mean([d['confidence'] for d in detections])

    return torch.tensor([
        furniture_count / 10.0,
        avg_area / 100.0,
        avg_confidence
    ], dtype=torch.float32)

# ============================================
# API ENDPOINTS
# ============================================

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "name": "PlaybookTV Interior Design AI",
        "version": "2.0.0",
        "status": "running",
        "models_loaded": {
            "yolo": models["yolo"] is not None,
            "style_ensemble": models["style_ensemble"] is not None
        },
        "device": models["device"],
        "endpoints": {
            "analyze": "/analyze",
            "detect": "/detect",
            "classify_style": "/classify/style",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models": {
            "yolo": "loaded" if models["yolo"] else "not_loaded",
            "style_ensemble": "loaded" if models["style_ensemble"] else "not_loaded"
        }
    }

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_image(file: UploadFile = File(...)):
    """
    Complete image analysis: object detection + style classification

    This is the main endpoint for the Modomo app.
    """
    start_time = datetime.now()

    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')

        # Run object detection
        detections = []
        if models["yolo"]:
            results = models["yolo"](image, verbose=False)

            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        class_name = models["yolo"].names[cls]

                        # Calculate area percentage
                        bbox_area = (x2 - x1) * (y2 - y1)
                        img_area = image.size[0] * image.size[1]
                        area_pct = (bbox_area / img_area) * 100

                        detections.append({
                            "item_type": class_name,
                            "confidence": conf,
                            "bbox": [x1, y1, x2, y2],
                            "area_percentage": area_pct
                        })

        # Run style classification
        style_result = None
        if models["style_ensemble"]:
            # Preprocess image
            image_tensor = preprocess_image(image).to(models["device"])

            # Extract context features
            context_features = extract_furniture_context(detections).unsqueeze(0).to(models["device"])

            # Predict with ensemble
            with torch.no_grad():
                probs = models["style_ensemble"].predict_ensemble(image_tensor, context_features)
                probs_np = probs.cpu().numpy()[0]

                style_idx = probs_np.argmax()
                style_name = config.STYLES[style_idx]
                confidence = float(probs_np[style_idx])

                all_probs = {
                    style: float(prob)
                    for style, prob in zip(config.STYLES, probs_np)
                }

                style_result = {
                    "style": style_name,
                    "confidence": confidence,
                    "all_probabilities": all_probs
                }

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return {
            "detections": detections,
            "detection_count": len(detections),
            "style": style_result or {
                "style": "unknown",
                "confidence": 0.0,
                "all_probabilities": {}
            },
            "processing_time_ms": processing_time
        }

    except Exception as e:
        logger.error(f"Error analyzing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    """
    Object detection only (faster)
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')

        if not models["yolo"]:
            raise HTTPException(status_code=503, detail="YOLO model not loaded")

        results = models["yolo"](image, verbose=False)

        detections = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = models["yolo"].names[cls]

                    bbox_area = (x2 - x1) * (y2 - y1)
                    img_area = image.size[0] * image.size[1]
                    area_pct = (bbox_area / img_area) * 100

                    detections.append({
                        "item_type": class_name,
                        "confidence": conf,
                        "bbox": [x1, y1, x2, y2],
                        "area_percentage": area_pct
                    })

        return {
            "detections": detections,
            "count": len(detections)
        }

    except Exception as e:
        logger.error(f"Error detecting objects: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify/style")
async def classify_style(file: UploadFile = File(...)):
    """
    Style classification only
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')

        if not models["style_ensemble"]:
            raise HTTPException(status_code=503, detail="Style classifier not loaded")

        # Preprocess
        image_tensor = preprocess_image(image).to(models["device"])
        context_features = torch.zeros(1, 3).to(models["device"])

        # Predict
        with torch.no_grad():
            probs = models["style_ensemble"].predict_ensemble(image_tensor, context_features)
            probs_np = probs.cpu().numpy()[0]

            style_idx = probs_np.argmax()
            style_name = config.STYLES[style_idx]
            confidence = float(probs_np[style_idx])

            all_probs = {
                style: float(prob)
                for style, prob in zip(config.STYLES, probs_np)
            }

        return {
            "style": style_name,
            "confidence": confidence,
            "all_probabilities": all_probs
        }

    except Exception as e:
        logger.error(f"Error classifying style: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/info")
async def model_info():
    """Get information about loaded models"""
    return {
        "yolo": {
            "loaded": models["yolo"] is not None,
            "path": config.YOLO_MODEL_PATH,
            "classes": len(models["yolo"].names) if models["yolo"] else 0
        },
        "style_ensemble": {
            "loaded": models["style_ensemble"] is not None,
            "models": ["efficientnet", "resnet50", "vit"],
            "styles": config.STYLES
        },
        "device": models["device"]
    }

# ============================================
# RUN SERVER
# ============================================

if __name__ == "__main__":
    import uvicorn

    # Configuration
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")

    logger.info(f"Starting server on {host}:{port}")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,  # Set to True for development
        log_level="info"
    )
