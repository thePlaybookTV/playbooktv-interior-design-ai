"""
PlaybookTV Interior Design AI - Simple API (No Models Required)
Use this for testing the API structure without trained models
"""

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from PIL import Image
import io
from datetime import datetime
import random

# Initialize FastAPI
app = FastAPI(
    title="PlaybookTV Interior Design AI (Mock)",
    description="API for testing - Returns mock predictions",
    version="2.0.0-mock"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

class AnalysisResult(BaseModel):
    detections: List[DetectionResult]
    detection_count: int
    style: StylePrediction
    processing_time_ms: float
    note: str = "Mock mode - using simulated predictions"

# Mock data
FURNITURE_TYPES = [
    "sectional_sofa", "coffee_table", "accent_chair", "floor_lamp",
    "area_rug", "dining_table", "dining_chair", "bookshelf",
    "tv_stand", "side_table", "pendant_light", "potted_plant"
]

STYLES = [
    'modern', 'traditional', 'contemporary', 'minimalist',
    'scandinavian', 'industrial', 'bohemian', 'mid_century_modern', 'rustic'
]

@app.get("/")
async def root():
    return {
        "name": "PlaybookTV Interior Design AI",
        "version": "2.0.0-mock",
        "status": "running",
        "mode": "MOCK - No real models loaded",
        "note": "Returns simulated predictions for testing",
        "endpoints": {
            "analyze": "/analyze",
            "detect": "/detect",
            "classify_style": "/classify/style",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "mode": "mock",
        "timestamp": datetime.now().isoformat(),
        "note": "API is running in mock mode"
    }

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_image(file: UploadFile = File(...)):
    """
    Mock analysis - returns random but realistic predictions
    """
    start_time = datetime.now()

    # Read image to validate it
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    width, height = image.size

    # Generate mock detections
    num_items = random.randint(2, 6)
    detections = []

    for i in range(num_items):
        item_type = random.choice(FURNITURE_TYPES)
        confidence = random.uniform(0.75, 0.95)

        # Random bbox
        x1 = random.randint(0, width - 200)
        y1 = random.randint(0, height - 200)
        x2 = x1 + random.randint(100, 300)
        y2 = y1 + random.randint(100, 300)

        bbox_area = (x2 - x1) * (y2 - y1)
        img_area = width * height
        area_pct = (bbox_area / img_area) * 100

        detections.append({
            "item_type": item_type,
            "confidence": confidence,
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "area_percentage": float(area_pct)
        })

    # Generate mock style prediction
    style_probs = {s: random.uniform(0, 1) for s in STYLES}
    total = sum(style_probs.values())
    style_probs = {s: p/total for s, p in style_probs.items()}

    top_style = max(style_probs.items(), key=lambda x: x[1])

    processing_time = (datetime.now() - start_time).total_seconds() * 1000

    return {
        "detections": detections,
        "detection_count": len(detections),
        "style": {
            "style": top_style[0],
            "confidence": top_style[1],
            "all_probabilities": style_probs
        },
        "processing_time_ms": processing_time,
        "note": "⚠️ MOCK MODE - These are simulated predictions for testing only"
    }

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    """Mock object detection"""
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    width, height = image.size

    num_items = random.randint(2, 6)
    detections = []

    for i in range(num_items):
        x1 = random.randint(0, width - 200)
        y1 = random.randint(0, height - 200)
        x2 = x1 + random.randint(100, 300)
        y2 = y1 + random.randint(100, 300)

        detections.append({
            "item_type": random.choice(FURNITURE_TYPES),
            "confidence": random.uniform(0.75, 0.95),
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "area_percentage": random.uniform(10, 30)
        })

    return {
        "detections": detections,
        "count": len(detections),
        "note": "Mock predictions"
    }

@app.post("/classify/style")
async def classify_style(file: UploadFile = File(...)):
    """Mock style classification"""
    style_probs = {s: random.uniform(0, 1) for s in STYLES}
    total = sum(style_probs.values())
    style_probs = {s: p/total for s, p in style_probs.items()}

    top_style = max(style_probs.items(), key=lambda x: x[1])

    return {
        "style": top_style[0],
        "confidence": top_style[1],
        "all_probabilities": style_probs,
        "note": "Mock prediction"
    }

@app.get("/models/info")
async def model_info():
    return {
        "mode": "mock",
        "yolo": {"loaded": False, "status": "mock mode"},
        "style_ensemble": {"loaded": False, "status": "mock mode"},
        "note": "No real models loaded - returning simulated predictions"
    }

if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.getenv("PORT", 8000))
    print(f"\n{'='*60}")
    print("🎭 MOCK MODE - API FOR TESTING ONLY")
    print(f"{'='*60}")
    print(f"\nStarting on http://0.0.0.0:{port}")
    print("This API returns simulated predictions for testing.")
    print("Train Phase 2 models for real predictions.")
    print(f"{'='*60}\n")

    uvicorn.run("main_simple:app", host="0.0.0.0", port=port, reload=True)
