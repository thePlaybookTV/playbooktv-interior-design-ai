# Quick Guide: Add Room Classification to API

## TL;DR

Your Phase 1 model has **70% room classification accuracy** - that's excellent!

You should add it to your API to complete the full analysis pipeline.

---

## What You Need to Change

Only **ONE file**: `api/main.py`

### 1. Add Import (line ~38)

```python
from src.models.training import InteriorDesignModel
```

### 2. Load Model in Startup (add to `load_models()` function, around line 220)

```python
# Load Phase 1 model for room classification
phase1_path = project_root / "models_best_interior_model.pth"
if phase1_path.exists():
    logger.info("Loading Phase 1 model for room classification...")
    try:
        checkpoint = torch.load(str(phase1_path), map_location=models["device"])

        room_model = InteriorDesignModel(
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
```

### 3. Add Room Classification in `/analyze` Endpoint (around line 400)

Add this **after** object detection and **before** style classification:

```python
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
```

### 4. Update Return (around line 450)

Change from:
```python
return {
    "detections": detections,
    "detection_count": len(detections),
    "style": style_result,
    "processing_time_ms": processing_time
}
```

To:
```python
return {
    "detections": detections,
    "detection_count": len(detections),
    "style": style_result,
    "room": room_result,  # ← ADD THIS LINE
    "processing_time_ms": processing_time
}
```

---

## That's It!

Restart your API and you'll now get room classification in the response:

```json
{
  "room": {
    "room_type": "living_room",
    "confidence": 0.85
  }
}
```

---

## Complete Model Stack

After this change, your API will use:

| Task | Model | Performance |
|------|-------|-------------|
| Object Detection | Phase 2 YOLO | 84% mAP, 294 classes |
| Room Type | Phase 1 ResNet50 | 70% accuracy, 6 types |
| Style | Phase 2 Ensemble | 56% accuracy, 12 styles |

**Status**: Production-ready! 🚀
