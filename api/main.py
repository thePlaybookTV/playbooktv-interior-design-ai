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

# Load .env file FIRST
from dotenv import load_dotenv
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

# Setup logging (after loading .env)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info(f"📋 Loaded environment from: {env_path}")

# Fix Python path to find src module
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import your models
try:
    from src.models.improved_style_classifier import EfficientNetStyleClassifier, EnsembleStyleClassifier
    from src.models.mask_enhanced_style_classifier import MaskEnhancedEnsemble
    from src.models.pristine_detector import PristineDetector
    from src.models.color_extractor import MaskBasedColorExtractor
    from src.models.shape_feature_extractor import ShapeFeatureExtractor
    from src.models.phase1_model import Phase1InteriorDesignModel
    from ultralytics import YOLO
    logger.info("✅ Model imports successful")
except ImportError as e:
    logger.warning(f"⚠️ Could not import models: {e}")
    logger.warning("Models will be loaded dynamically when available")
    EfficientNetStyleClassifier = None
    EnsembleStyleClassifier = None
    MaskEnhancedEnsemble = None
    PristineDetector = None
    MaskBasedColorExtractor = None
    ShapeFeatureExtractor = None
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
    "pristine_detector": None,  # YOLO + SAM2
    "room_classifier": None,  # Phase 1 room classification
    "style_ensemble": None,
    "mask_enhanced_ensemble": None,  # Mask-enhanced style classifier
    "color_extractor": None,
    "shape_extractor": None,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# Configuration
class Config:
    # Model paths (configure these)
    YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "./models/yolo_best.pt")
    EFFICIENTNET_PATH = os.getenv("EFFICIENTNET_PATH", "./models/best_efficientnet_style_classifier.pth")
    RESNET_PATH = os.getenv("RESNET_PATH", "./models/best_resnet_style_classifier.pth")
    VIT_PATH = os.getenv("VIT_PATH", "./models/best_vit_style_classifier.pth")

    # Mask-enhanced models
    MASK_ENHANCED_DIR = os.getenv("MASK_ENHANCED_DIR", "./mask_enhanced_models")
    DB_PATH = os.getenv("DB_PATH", "./database_metadata.duckdb")

    # Use SAM2 enhanced detection
    USE_SAM2 = os.getenv("USE_SAM2", "true").lower() == "true"

    # Categories
    ROOM_TYPES = ['living_room', 'bedroom', 'kitchen', 'dining_room', 'bathroom', 'home_office']
    # Phase 2 trained with 12 styles from the database
    STYLES = ['bohemian', 'coastal', 'contemporary', 'eclectic', 'industrial',
              'mid_century_modern', 'minimalist', 'modern', 'rustic', 'scandinavian',
              'traditional', 'transitional']

config = Config()

# Response models
class DetectionResult(BaseModel):
    item_type: str
    confidence: float
    bbox: List[float]
    area_percentage: float
    has_mask: Optional[bool] = False
    mask_score: Optional[float] = None
    mask_area: Optional[int] = None
    colors: Optional[Dict] = None  # Per-furniture colors

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
        # Load YOLO model (Phase 2 fine-tuned or PristineDetector)
        if config.USE_SAM2:
            # Use PristineDetector (YOLO + SAM2) for segmentation masks
            logger.info("Loading PristineDetector (YOLO + SAM2)...")
            try:
                models["pristine_detector"] = PristineDetector()
                logger.info("✅ PristineDetector loaded (YOLO + SAM2)")
            except Exception as e:
                logger.warning(f"⚠️ Could not load PristineDetector: {e}")
                logger.info("Falling back to YOLO-only mode")
                config.USE_SAM2 = False

        # Load YOLO standalone (Phase 2 fine-tuned model or fallback)
        if not config.USE_SAM2:
            yolo_path = Path(config.YOLO_MODEL_PATH)
            if yolo_path.exists():
                logger.info(f"Loading Phase 2 YOLO from {config.YOLO_MODEL_PATH}")
                try:
                    models["yolo"] = YOLO(config.YOLO_MODEL_PATH)
                    logger.info(f"✅ YOLO loaded successfully ({len(models['yolo'].names)} classes)")
                except Exception as e:
                    logger.error(f"❌ Failed to load YOLO: {e}")
                    raise
            else:
                logger.error(f"❌ YOLO model not found at {config.YOLO_MODEL_PATH}")
                logger.error(f"   Please verify the path in .env file")
                raise FileNotFoundError(f"YOLO model not found: {config.YOLO_MODEL_PATH}")

        # Load Mask-Enhanced Style Ensemble (if available)
        mask_enhanced_dir = Path(config.MASK_ENHANCED_DIR)
        if mask_enhanced_dir.exists():
            logger.info("Loading Mask-Enhanced Style Ensemble...")
            try:
                ensemble = MaskEnhancedEnsemble(
                    num_styles=len(config.STYLES),
                    device=models["device"]
                )
                ensemble.load_ensemble(str(mask_enhanced_dir))

                # Set to eval mode
                for model in ensemble.models:
                    model.eval()

                models["mask_enhanced_ensemble"] = ensemble
                logger.info("✅ Mask-Enhanced Style Ensemble loaded")
            except Exception as e:
                logger.warning(f"⚠️ Could not load mask-enhanced ensemble: {e}")

        # Load Standard Style Ensemble (fallback)
        if models["mask_enhanced_ensemble"] is None:
            if all(Path(p).exists() for p in [config.EFFICIENTNET_PATH, config.RESNET_PATH, config.VIT_PATH]):
                logger.info("Loading Standard Style Ensemble...")
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
                logger.info("✅ Standard Style Ensemble loaded")
            else:
                logger.warning("⚠️ Style classifier models not found")

        # Load feature extractors
        if Path(config.DB_PATH).exists():
            logger.info("Loading feature extractors...")
            models["color_extractor"] = MaskBasedColorExtractor(config.DB_PATH)
            models["shape_extractor"] = ShapeFeatureExtractor(config.DB_PATH)
            logger.info("✅ Feature extractors loaded")

        # Load Phase 1 model for room classification
        phase1_path = project_root / "models_best_interior_model.pth"
        if phase1_path.exists():
            logger.info("Loading Phase 1 model for room classification...")
            try:
                checkpoint = torch.load(str(phase1_path), map_location=models["device"])

                room_model = Phase1InteriorDesignModel(
                    num_rooms=len(checkpoint['room_types']),
                    num_styles=len(checkpoint['styles'])
                )
                room_model.load_state_dict(checkpoint['model_state_dict'])
                room_model.eval()
                room_model.to(models["device"])

                models["room_classifier"] = room_model
                config.ROOM_TYPES = checkpoint['room_types']

                logger.info(f"✅ Room classifier loaded (6 types, 70% accuracy)")
            except Exception as e:
                logger.warning(f"⚠️ Could not load room classifier: {e}")

        logger.info(f"🎮 Using device: {models['device']}")
        logger.info(f"🎭 SAM2 enabled: {config.USE_SAM2}")
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
        "version": "3.0.0",
        "status": "running",
        "features": {
            "sam2_masks": config.USE_SAM2,
            "mask_enhanced_classifier": models["mask_enhanced_ensemble"] is not None,
            "color_extraction": models["color_extractor"] is not None,
            "shape_features": models["shape_extractor"] is not None
        },
        "models_loaded": {
            "pristine_detector": models["pristine_detector"] is not None,
            "yolo": models["yolo"] is not None,
            "mask_enhanced_ensemble": models["mask_enhanced_ensemble"] is not None,
            "style_ensemble": models["style_ensemble"] is not None
        },
        "device": models["device"],
        "endpoints": {
            "analyze": "/analyze - Full analysis with SAM2 masks + style",
            "detect": "/detect - Object detection only",
            "classify_style": "/classify/style - Style classification only",
            "health": "/health - Health check",
            "models_info": "/models/info - Model information"
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

        if config.USE_SAM2 and models["pristine_detector"]:
            # Use YOLO + SAM2 detection with masks
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                image.save(tmp.name)
                result = models["pristine_detector"].detect_with_masks(tmp.name)
                os.unlink(tmp.name)

            # Convert PIL image to numpy for color extraction
            image_np = np.array(image)

            for item in result.get('items', []):
                detection = {
                    "item_type": item['type'],
                    "confidence": item['confidence'],
                    "bbox": item['bbox'],
                    "area_percentage": item['area_percentage'],
                    "has_mask": item.get('has_mask', False),
                    "mask_score": item.get('mask_score'),
                    "mask_area": item.get('mask_area')
                }

                # Extract colors if mask available and color extractor loaded
                if item.get('has_mask') and item.get('mask') is not None and models["color_extractor"]:
                    try:
                        # Extract colors from actual SAM2 mask
                        mask_array = np.array(item['mask'])
                        color_features = models["color_extractor"].extract_colors_from_mask(
                            image_np, mask_array, n_colors=3
                        )
                        detection['colors'] = {
                            'palette': color_features.get('palette', []),
                            'dominant': color_features.get('dominant_color', None)
                        }
                    except Exception as e:
                        logger.warning(f"Color extraction failed for {item['type']}: {e}")

                detections.append(detection)

        elif models["yolo"]:
            # Fallback to YOLO-only detection
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
                            "area_percentage": area_pct,
                            "has_mask": False
                        })

        # Run room classification
        room_result = None
        if models.get("room_classifier"):
            try:
                image_tensor = preprocess_image(image).to(models["device"])
                furniture_features = extract_furniture_context(detections).unsqueeze(0).to(models["device"])

                with torch.no_grad():
                    room_logits, _ = models["room_classifier"](image_tensor, furniture_features)
                    room_probs = torch.softmax(room_logits, dim=1)[0]
                    room_idx = room_probs.argmax().item()

                    room_result = {
                        "room_type": config.ROOM_TYPES[room_idx],
                        "confidence": float(room_probs[room_idx])
                    }
            except Exception as e:
                logger.error(f"Room classification error: {e}")

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
            "room": room_result,
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
