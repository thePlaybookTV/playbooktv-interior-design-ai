# Model Consolidation Guide

## Overview

You have **two different model architectures** from different training phases. Here's how they relate and which to use.

---

## Model Comparison

### Phase 1 Model: `models_best_interior_model.pth` (130 MB)

**Architecture**: Single multi-task ResNet50 model
- **Purpose**: Joint room type + style classification
- **Input**: Image only (no furniture context)
- **Outputs**:
  - Room type (e.g., living_room, bedroom, kitchen)
  - Interior style (e.g., modern, traditional, bohemian)

**Training Approach**:
- Single ResNet50 backbone shared between tasks
- Trained end-to-end on images with room/style labels
- Uses furniture count as additional feature
- File: `src/models/training.py` → `InteriorDesignModel`

**Saved Checkpoint Contains**:
```python
{
    'epoch': int,
    'model_state_dict': OrderedDict,  # ResNet50 weights
    'optimizer_state_dict': OrderedDict,
    'val_loss': float,
    'val_room_acc': float,  # ~53.8% typical
    'val_style_acc': float,  # ~53.8% typical
    'room_types': list,     # Room categories
    'styles': list          # Style categories
}
```

---

### Phase 2 Models: Ensemble (3 separate files, ~500 MB total)

**Architecture**: Ensemble of 3 specialized classifiers
- **Purpose**: Style classification ONLY (improved accuracy)
- **Input**: Image + furniture context features
- **Output**: Interior style (12 categories)

**Models**:
1. `best_efficientnet_style_classifier.pth` - EfficientNet-B0 backbone
2. `best_resnet_style_classifier.pth` - ResNet50 backbone
3. `best_vit_style_classifier.pth` - Vision Transformer backbone

**Training Approach**:
- Each model trained independently
- **Ensemble voting** for final prediction
- Uses furniture detection context (count, area, confidence)
- File: `src/models/improved_style_classifier.py` → `EnsembleStyleClassifier`

**Performance**: 56.48% accuracy (2.68% improvement over Phase 1)

**Key Difference**: These models are **style-only**, not multi-task like Phase 1

---

## Current API Configuration (Production)

Your API currently uses:

### Object Detection
✅ **Phase 2 Fine-tuned YOLO** (`yolo_training_runs/finetune_294_classes/weights/best.pt`)
- 294 interior design categories
- 84.15% mAP50
- Trained specifically on your dataset

### Style Classification
✅ **Phase 2 Ensemble** (3 models)
- EfficientNet + ResNet50 + ViT
- 56.48% accuracy
- 12 style categories

### Room Type Classification
❌ **MISSING** - No room classifier loaded!

---

## The Problem: Missing Room Classification

**Phase 1 model** does room + style (but lower accuracy)
**Phase 2 models** do style only (higher accuracy but no room type)

### Current API Behavior:
```python
@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    # ✅ Object detection works (YOLO)
    # ✅ Style classification works (Ensemble)
    # ❌ Room type classification: RETURNS None
```

---

## Solution Options

### Option 1: Use Phase 1 Model for Room Type Only (Recommended)

**Keep Phase 2 for style, add Phase 1 for room classification**

```python
# api/main.py

# Load Phase 1 model for room type
if Path("models_best_interior_model.pth").exists():
    phase1_checkpoint = torch.load("models_best_interior_model.pth")
    room_classifier = InteriorDesignModel(
        num_rooms=len(phase1_checkpoint['room_types']),
        num_styles=len(phase1_checkpoint['styles'])
    )
    room_classifier.load_state_dict(phase1_checkpoint['model_state_dict'])
    room_classifier.eval()
    models["room_classifier"] = room_classifier
    models["room_types"] = phase1_checkpoint['room_types']
```

**Then in `/analyze` endpoint**:
```python
# Detect room type with Phase 1 model
if models["room_classifier"]:
    image_tensor = preprocess_image(image).to(device)
    furniture_features = extract_furniture_context(detections)

    with torch.no_grad():
        room_logits, _ = models["room_classifier"](image_tensor, furniture_features)
        room_probs = torch.softmax(room_logits, dim=1)
        room_idx = room_probs.argmax(1).item()
        room_confidence = room_probs[0, room_idx].item()

    room_result = {
        "room_type": models["room_types"][room_idx],
        "confidence": room_confidence
    }
```

**Pros**:
- Gets you room classification immediately
- Best of both worlds: Phase 2 style accuracy + Phase 1 room classification
- Simple to implement

**Cons**:
- Running two models (slight performance overhead)
- Room accuracy still ~53.8%

---

### Option 2: Train New Room Classifier (Phase 3)

Train a dedicated room type classifier similar to the Phase 2 style ensemble.

**Would require**:
1. Copy `src/models/improved_style_classifier.py` → `improved_room_classifier.py`
2. Change output to room types instead of styles
3. Train ensemble on room classification task
4. Save 3 new models

**Pros**:
- Best possible room classification accuracy
- Clean separation of concerns

**Cons**:
- Requires more training time (~30-60 minutes)
- Another 500 MB of model files
- Total of 6 classifier models in production

---

### Option 3: Hybrid - Use CLIP for Room Type (Fast)

The `ImageProcessor` already uses CLIP for zero-shot classification. You could use it in the API.

**Pros**:
- No additional model files
- CLIP already loaded in processing pipeline
- Fast inference

**Cons**:
- CLIP accuracy may be lower than trained model
- Not optimized for your specific dataset

---

## Recommended Architecture

```
API Inference Pipeline
│
├─ Object Detection
│  └─ Phase 2 YOLO (294 classes, 84% mAP)
│
├─ Room Classification
│  └─ Phase 1 Model (room head only, ~54% acc) ← ADD THIS
│
└─ Style Classification
   └─ Phase 2 Ensemble (12 styles, 56% acc)
```

**File locations**:
- Phase 1: `models_best_interior_model.pth` (already exists)
- Phase 2 YOLO: `yolo_training_runs/finetune_294_classes/weights/best.pt` (✅ loaded)
- Phase 2 Style: `best_*_style_classifier.pth` (✅ loaded)

---

## Implementation Plan

### Step 1: Import Phase 1 Model Class

Add to `api/main.py`:
```python
from src.models.training import InteriorDesignModel
```

### Step 2: Load Phase 1 Model in Startup

Add to `load_models()` function:
```python
# Load Phase 1 model for room classification
phase1_path = Path("models_best_interior_model.pth")
if phase1_path.exists():
    logger.info("Loading Phase 1 model for room classification...")
    checkpoint = torch.load(str(phase1_path), map_location=models["device"])

    room_model = InteriorDesignModel(
        num_rooms=len(checkpoint['room_types']),
        num_styles=len(checkpoint['styles'])
    )
    room_model.load_state_dict(checkpoint['model_state_dict'])
    room_model.eval()

    models["room_classifier"] = room_model
    config.ROOM_TYPES = checkpoint['room_types']
    logger.info(f"✅ Room classifier loaded ({len(config.ROOM_TYPES)} types)")
```

### Step 3: Update `/analyze` Endpoint

Add room classification after object detection:
```python
# Run room classification
room_result = None
if models["room_classifier"]:
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
```

### Step 4: Update Response Model

The `AnalysisResult` model already has `room: Optional[RoomPrediction]`, so just pass it:
```python
return {
    "detections": detections,
    "detection_count": len(detections),
    "style": style_result,
    "room": room_result,  # ← Will now be populated
    "processing_time_ms": processing_time
}
```

---

## Summary

**Current State**:
- ✅ Object Detection: Phase 2 YOLO (excellent)
- ✅ Style Classification: Phase 2 Ensemble (excellent)
- ❌ Room Classification: Missing

**Recommendation**:
Use **Option 1** - Load Phase 1 model for room classification while keeping Phase 2 for everything else.

**Why**:
- Fastest to implement (no new training)
- Gets you full functionality immediately
- Best accuracy combination available
- Can always train Phase 3 room classifier later if needed

**Future Enhancement** (Optional):
Train dedicated room classifier ensemble (Phase 3) for even better room accuracy.

---

## Model File Reference

```
playbooktv-interior-design-ai/
│
├── models_best_interior_model.pth                    # Phase 1 (130 MB)
│   └─ Multi-task: Room + Style
│
├── best_efficientnet_style_classifier.pth            # Phase 2 (1/3)
├── best_resnet_style_classifier.pth                  # Phase 2 (2/3)
├── best_vit_style_classifier.pth                     # Phase 2 (3/3)
│   └─ Ensemble: Style only
│
└── yolo_training_runs/finetune_294_classes/weights/
    └── best.pt                                        # Phase 2 YOLO
        └─ Object Detection: 294 classes
```

**Total Model Size**: ~630 MB (acceptable for production)
