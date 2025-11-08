# Phase 2 Quick Start Guide

## What's New in Phase 2?

Phase 2 introduces **two major upgrades** to improve the PlaybookTV Interior Design AI:

### 1. YOLO Fine-tuning (294 Categories)
- **Before**: 14 generic COCO classes (chair, couch, table, etc.)
- **After**: 294 specific interior design categories (wingback_chair, sectional_sofa, coffee_table, etc.)
- **Benefit**: Much more detailed and accurate object detection

### 2. Improved Style Classification (70%+ Accuracy)
- **Before**: Single ResNet18 model with 53.8% accuracy
- **After**: Ensemble of 3 models (EfficientNet + ResNet50 + ViT) with 70%+ accuracy
- **Benefit**: +16-21% improvement in style classification accuracy

---

## Prerequisites

✅ **Phase 1 must be completed first**
- Database populated with images and detections
- At least 5,000 images processed

✅ **Hardware Requirements**
- NVIDIA GPU with 8GB+ VRAM (16GB recommended)
- 16GB+ RAM
- 50GB free storage

✅ **Software Requirements**
```bash
pip install ultralytics  # For YOLOv8
```

---

## Option 1: Run Everything (Recommended)

Train both YOLO and Style Classification in one go:

```bash
python scripts/run_phase2_training.py \
    --db ./interior_design_data_hybrid/processed/metadata.duckdb \
    --output ./phase2_outputs \
    --yolo-epochs 100 \
    --style-epochs 30 \
    --batch-size 16
```

**Expected Time**: 10-16 hours on A4000 GPU

**What You Get**:
- Fine-tuned YOLO model (294 classes)
- Ensemble style classifier (3 models)
- Comprehensive training report

---

## Option 2: YOLO Only

Just fine-tune YOLO on 294 categories:

```bash
python scripts/run_phase2_training.py \
    --skip-style \
    --yolo-epochs 100 \
    --batch-size 16
```

**Expected Time**: 8-12 hours on A4000 GPU

---

## Option 3: Style Classification Only

Just train the improved style classifier:

```bash
python scripts/run_phase2_training.py \
    --skip-yolo \
    --style-epochs 30 \
    --batch-size 32
```

**Expected Time**: 2-4 hours on A4000 GPU

---

## Manual Step-by-Step

### Step 1: Prepare YOLO Dataset

```python
from src.models.yolo_dataset_prep import YOLODatasetBuilder

builder = YOLODatasetBuilder(
    db_path="./interior_design_data_hybrid/processed/metadata.duckdb",
    output_dir="./yolo_dataset"
)

result = builder.prepare_dataset(train_split=0.8, min_confidence=0.5)
builder.close()
```

### Step 2: Train YOLO

```bash
python src/models/yolo_finetune.py \
    --data ./yolo_dataset/data.yaml \
    --model yolov8m.pt \
    --epochs 100 \
    --batch 16 \
    --mode train
```

### Step 3: Train Style Classifier

```python
from src.models.improved_style_classifier import train_improved_style_classifier

ensemble, results, accuracy = train_improved_style_classifier(
    db_path="./interior_design_data_hybrid/processed/metadata.duckdb",
    epochs=30,
    batch_size=32
)
```

---

## Expected Results

### YOLO Object Detection

| Metric | Target |
|--------|--------|
| mAP50 | 0.70+ |
| mAP50-95 | 0.50+ |
| Precision | 0.75+ |
| Recall | 0.70+ |

### Style Classification

| Model | Expected Accuracy |
|-------|-------------------|
| EfficientNet | ~68% |
| ResNet50 | ~65% |
| ViT-B/16 | ~63% |
| **Ensemble** | **70-75%** |

**Improvement over Phase 1**: +16-21% (from 53.8% to 70%+)

---

## Output Files

After training, you'll find:

```
phase2_outputs/
├── yolo_dataset/                       # Prepared YOLO dataset
│   ├── data.yaml
│   ├── category_mapping.json
│   └── images/ & labels/
├── yolo_training_runs/                 # YOLO training outputs
│   └── finetune_294_classes/
│       ├── weights/best.pt             # Best YOLO model ⭐
│       └── results.csv
├── best_efficientnet_style_classifier.pth  # Best EfficientNet ⭐
├── best_resnet_style_classifier.pth        # Best ResNet50 ⭐
├── best_vit_style_classifier.pth           # Best ViT ⭐
└── phase2_report_[timestamp].json      # Final report ⭐
```

---

## Testing Your Models

### Test YOLO Detection

```bash
python src/models/yolo_finetune.py \
    --mode test \
    --weights ./phase2_outputs/yolo_training_runs/finetune_294_classes/weights/best.pt \
    --test-image /path/to/test/image.jpg
```

### Test Style Classification

```python
from src.models.improved_style_classifier import EnsembleStyleClassifier
import torch

# Load ensemble
ensemble = EnsembleStyleClassifier(num_styles=9, device='cuda')

# Load individual model weights
ensemble.models['efficientnet'].load_state_dict(
    torch.load('best_efficientnet_style_classifier.pth')
)
# ... load other models ...

# Predict on new image
# predictions = ensemble.predict_ensemble(image_tensor, context_features)
```

---

## Troubleshooting

### CUDA Out of Memory

**Solution 1**: Reduce batch size
```bash
--batch-size 8  # or even 4
```

**Solution 2**: Use smaller YOLO model
```bash
--model yolov8s.pt  # instead of yolov8m.pt
```

**Solution 3**: Reduce image size
```bash
--imgsz 512  # instead of 640
```

### Low Accuracy

**YOLO**:
- Check `category_mapping.json` for correct class mappings
- Ensure at least 100 examples per category
- Try training for more epochs

**Style Classification**:
- Check for class imbalance in dataset
- Increase augmentation strength
- Train for more epochs (30 → 50)

---

## Monitoring Training

### YOLO Training

Training metrics are automatically logged:
- Real-time progress bar
- Loss curves (box, cls, dfl)
- mAP metrics
- Results saved to `results.csv`

### Style Classification

Each model shows:
- Per-epoch train/val accuracy
- Best model checkpointing
- Final ensemble evaluation

---

## Next Steps After Phase 2

1. **Evaluate on Test Set**
   - Collect new unseen images
   - Measure real-world performance

2. **Deploy Models**
   - Export YOLO to ONNX for production
   - Create FastAPI inference endpoint

3. **Iterate and Improve**
   - Analyze failure cases
   - Collect more data for weak categories
   - Fine-tune ensemble weights

4. **Phase 3 Planning**
   - Real-time inference API
   - Mobile app integration
   - 3D room reconstruction
   - Style recommendation engine

---

## Support

**For detailed documentation**: See [PHASE2_GUIDE.md](docs/PHASE2_GUIDE.md)

**Common Issues**:
- GPU not detected: `nvidia-smi` to check
- Missing dependencies: `pip install -r requirements.txt`
- Database errors: Verify Phase 1 completed successfully

---

## Key Differences: Phase 1 vs Phase 2

| Feature | Phase 1 | Phase 2 |
|---------|---------|---------|
| **Object Detection** | 14 COCO classes | 294 custom categories |
| **Detection Model** | YOLOv8m (pretrained) | YOLOv8m (fine-tuned) |
| **Style Classification** | ResNet18 (53.8%) | Ensemble (70%+) |
| **Augmentation** | Basic | Advanced |
| **Context Features** | No | Yes (furniture-aware) |
| **Model Architectures** | 1 (ResNet18) | 3 (EfficientNet, ResNet50, ViT) |

---

**Ready to start? Run this command:**

```bash
python scripts/run_phase2_training.py --help
```

Good luck! 🚀
