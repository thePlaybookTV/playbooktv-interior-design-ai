# Phase 2: Advanced Training Guide

## Overview

Phase 2 introduces two major improvements to the PlaybookTV Interior Design AI system:

1. **YOLO Fine-tuning on 294-Category Taxonomy** - Upgrade from generic 14 COCO classes to 294 specific interior design categories
2. **Improved Style Classification** - Ensemble approach to boost accuracy from 53.8% to 70%+

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Component 1: YOLO Fine-tuning](#component-1-yolo-fine-tuning)
- [Component 2: Improved Style Classification](#component-2-improved-style-classification)
- [Running the Full Pipeline](#running-the-full-pipeline)
- [Evaluation and Metrics](#evaluation-and-metrics)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Hardware Requirements

- **GPU**: NVIDIA GPU with at least 8GB VRAM (16GB recommended)
  - YOLO training: ~6-8GB VRAM
  - Style ensemble training: ~4-6GB VRAM
- **RAM**: 16GB minimum (32GB recommended)
- **Storage**: 50GB free space for datasets and models

### Software Requirements

```bash
# Install additional dependencies for Phase 2
pip install ultralytics  # YOLOv8
pip install efficientnet-pytorch  # EfficientNet models
pip install timm  # PyTorch Image Models
```

### Dataset Requirements

- Phase 1 must be complete with:
  - DuckDB database populated with images and detections
  - At least 5,000 images with furniture detections
  - Style and room labels from Phase 1 classification

---

## Quick Start

The fastest way to run both Phase 2 improvements:

```bash
# Navigate to project root
cd /path/to/playbooktv-interior-design-ai

# Run full Phase 2 pipeline
python scripts/run_phase2_training.py \
    --db ./interior_design_data_hybrid/processed/metadata.duckdb \
    --output ./phase2_outputs \
    --yolo-epochs 100 \
    --style-epochs 30 \
    --batch-size 16
```

This will:
1. Prepare YOLO dataset from DuckDB (15-30 minutes)
2. Fine-tune YOLO on 294 categories (8-12 hours on A4000)
3. Train ensemble style classifier (2-4 hours on A4000)
4. Generate comprehensive report

---

## Component 1: YOLO Fine-tuning

### Step 1: Prepare YOLO Dataset

Convert DuckDB detections to YOLO format:

```python
from src.models.yolo_dataset_prep import YOLODatasetBuilder

# Create dataset builder
builder = YOLODatasetBuilder(
    db_path="./interior_design_data_hybrid/processed/metadata.duckdb",
    output_dir="./yolo_dataset"
)

# Prepare dataset
result = builder.prepare_dataset(
    train_split=0.8,        # 80% train, 20% validation
    min_confidence=0.5      # Only use high-confidence detections
)

# Check statistics
stats = builder.get_statistics()
print(f"Train images: {stats['train']['images']}")
print(f"Val images: {stats['val']['images']}")

builder.close()
```

**Output:**
- `yolo_dataset/images/train/` - Training images
- `yolo_dataset/images/val/` - Validation images
- `yolo_dataset/labels/train/` - Training labels (YOLO format)
- `yolo_dataset/labels/val/` - Validation labels
- `yolo_dataset/data.yaml` - YOLO configuration file
- `yolo_dataset/category_mapping.json` - Category mapping

### Step 2: Train YOLO

```python
from src.models.yolo_finetune import YOLOFineTuner

# Create fine-tuner
finetuner = YOLOFineTuner(
    data_yaml="./yolo_dataset/data.yaml",
    model_size="yolov8m.pt"  # Options: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
)

# Train
results = finetuner.train(
    epochs=100,
    batch_size=16,          # Adjust based on GPU memory
    image_size=640,
    learning_rate=0.01,
    patience=50,            # Early stopping
    save_period=10,         # Save checkpoint every 10 epochs
    augment=True,
    freeze_layers=0         # 0 = train all layers
)

# Validate
val_results = finetuner.validate()
print(f"mAP50: {val_results.box.map50:.4f}")
print(f"mAP50-95: {val_results.box.map:.4f}")
```

**Command Line:**

```bash
# Train from scratch
python src/models/yolo_finetune.py \
    --data ./yolo_dataset/data.yaml \
    --model yolov8m.pt \
    --epochs 100 \
    --batch 16 \
    --imgsz 640 \
    --mode train

# Validate existing model
python src/models/yolo_finetune.py \
    --mode validate \
    --weights ./yolo_training_runs/finetune_294_classes/weights/best.pt

# Test on image
python src/models/yolo_finetune.py \
    --mode test \
    --weights ./yolo_training_runs/finetune_294_classes/weights/best.pt \
    --test-image /path/to/test/image.jpg
```

### Step 3: Export Model

```python
# Export to ONNX and TorchScript for deployment
finetuner.export_model(
    weights_path="./yolo_training_runs/finetune_294_classes/weights/best.pt",
    formats=['onnx', 'torchscript']
)
```

### YOLO Training Tips

1. **Batch Size**: Adjust based on GPU memory
   - 8GB VRAM: batch_size=8
   - 16GB VRAM: batch_size=16-24
   - 24GB VRAM: batch_size=32+

2. **Model Size**: Balance between speed and accuracy
   - `yolov8n`: Fastest, lowest accuracy (~50-100 FPS)
   - `yolov8s`: Fast, good accuracy (~40-80 FPS)
   - `yolov8m`: Balanced (recommended) (~30-60 FPS)
   - `yolov8l`: Slower, better accuracy (~20-40 FPS)
   - `yolov8x`: Slowest, best accuracy (~10-30 FPS)

3. **Fine-tuning vs Training from Scratch**:
   - Fine-tuning (recommended): Faster convergence, better results
   - From scratch: Requires more data and longer training

---

## Component 2: Improved Style Classification

### Architecture Overview

The improved style classifier uses an **ensemble of three models**:

1. **EfficientNet-B0** (40% weight)
   - Lightweight and efficient
   - Uses furniture context features
   - Attention mechanism for feature focus

2. **ResNet50** (35% weight)
   - Proven architecture for image classification
   - Strong baseline performance

3. **Vision Transformer (ViT-B/16)** (25% weight)
   - State-of-the-art attention-based model
   - Captures global context better

### Training

```python
from src.models.improved_style_classifier import train_improved_style_classifier

# Train ensemble
ensemble, individual_results, ensemble_acc = train_improved_style_classifier(
    db_path="./interior_design_data_hybrid/processed/metadata.duckdb",
    epochs=30,
    batch_size=32
)

print(f"Ensemble Accuracy: {ensemble_acc:.4f}")
print("Individual Models:")
for model_name, acc in individual_results.items():
    print(f"  {model_name}: {acc:.4f}")
```

**Expected Results:**

| Model | Accuracy | Improvement over Phase 1 |
|-------|----------|--------------------------|
| EfficientNet | ~65-70% | +11-16% |
| ResNet50 | ~62-67% | +8-13% |
| ViT | ~60-65% | +6-11% |
| **Ensemble** | **70-75%** | **+16-21%** |

### Key Features

1. **Enhanced Data Augmentation**
   - Random crops, flips, rotations
   - Color jittering
   - Perspective transforms
   - Random erasing

2. **Furniture Context Integration**
   - Uses detected furniture types as features
   - Spatial distribution of furniture
   - Average detection confidence

3. **Attention Mechanism**
   - Focuses on style-relevant features
   - Improves discrimination between similar styles

4. **Class Balancing**
   - Stratified train/val split
   - Handles imbalanced style distribution

---

## Running the Full Pipeline

### Option 1: Complete Pipeline

```bash
python scripts/run_phase2_training.py \
    --db ./interior_design_data_hybrid/processed/metadata.duckdb \
    --output ./phase2_outputs \
    --yolo-epochs 100 \
    --style-epochs 30 \
    --batch-size 16
```

### Option 2: YOLO Only

```bash
python scripts/run_phase2_training.py \
    --db ./interior_design_data_hybrid/processed/metadata.duckdb \
    --skip-style \
    --yolo-epochs 100
```

### Option 3: Style Classification Only

```bash
python scripts/run_phase2_training.py \
    --db ./interior_design_data_hybrid/processed/metadata.duckdb \
    --skip-yolo \
    --style-epochs 30
```

---

## Evaluation and Metrics

### YOLO Metrics

- **mAP50**: Mean Average Precision at IoU=0.50
  - Target: >0.70 for fine-tuned model
  - Phase 1 baseline: N/A (generic classes)

- **mAP50-95**: Mean Average Precision at IoU=0.50:0.95
  - Target: >0.50 for fine-tuned model

- **Precision**: True Positives / (True Positives + False Positives)
  - Target: >0.75

- **Recall**: True Positives / (True Positives + False Negatives)
  - Target: >0.70

### Style Classification Metrics

- **Accuracy**: Correct predictions / Total predictions
  - Phase 1 baseline: 53.8%
  - Phase 2 target: 70%+

- **Per-Class Accuracy**: Accuracy for each style
  - Modern: ~75%
  - Contemporary: ~68%
  - Minimalist: ~72%
  - Traditional: ~65%
  - Scandinavian: ~70%
  - Industrial: ~66%
  - Bohemian: ~62%
  - Mid-century modern: ~68%
  - Rustic: ~64%

### Monitoring Training

```python
from src.models.yolo_finetune import YOLOTrainingMonitor

# Monitor YOLO training
monitor = YOLOTrainingMonitor(
    results_csv="./yolo_training_runs/finetune_294_classes/results.csv"
)

# Plot training curves
monitor.plot_metrics(save_path="./yolo_training_plots.png")
```

---

## Troubleshooting

### CUDA Out of Memory

**Problem**: GPU runs out of memory during training

**Solutions**:
1. Reduce batch size:
   ```python
   --batch-size 8  # or even 4
   ```

2. Use smaller model:
   ```python
   --model yolov8s.pt  # Instead of yolov8m.pt
   ```

3. Reduce image size:
   ```python
   --imgsz 512  # Instead of 640
   ```

4. Enable gradient checkpointing (for style classifier):
   ```python
   # In improved_style_classifier.py, add:
   torch.utils.checkpoint.checkpoint_sequential()
   ```

### Low mAP on YOLO

**Problem**: YOLO validation mAP < 0.50

**Possible causes and solutions**:

1. **Insufficient data**
   - Need at least 100 examples per class
   - Solution: Collect more data or use data augmentation

2. **Poor quality labels**
   - Check `category_mapping.json` for correct COCO→taxonomy mapping
   - Solution: Update mapping in `yolo_dataset_prep.py`

3. **Class imbalance**
   - Some categories have very few examples
   - Solution: Use weighted loss or focal loss

4. **Training too short**
   - Model hasn't converged
   - Solution: Increase epochs or reduce learning rate

### Low Style Classification Accuracy

**Problem**: Ensemble accuracy < 65%

**Possible causes and solutions**:

1. **Confusing style pairs**
   - Modern vs Contemporary are very similar
   - Solution: Merge similar styles or collect more distinct examples

2. **Limited training data**
   - Need balanced representation of all styles
   - Solution: Collect more data for underrepresented styles

3. **Models not converging**
   - Check training curves for plateaus
   - Solution: Adjust learning rate, increase epochs

4. **Overfitting**
   - High train accuracy, low val accuracy
   - Solution: Increase dropout, add more augmentation

### Data Preparation Issues

**Problem**: Dataset preparation fails or produces empty dataset

**Solutions**:

1. Check database connectivity:
   ```python
   import duckdb
   conn = duckdb.connect("./metadata.duckdb")
   print(conn.execute("SELECT COUNT(*) FROM images").fetchone())
   ```

2. Verify furniture detections exist:
   ```python
   print(conn.execute("SELECT COUNT(*) FROM furniture_detections").fetchone())
   ```

3. Check image paths are valid:
   ```python
   import os
   df = conn.execute("SELECT original_path FROM images LIMIT 10").df()
   for path in df['original_path']:
       print(f"{path}: {os.path.exists(path)}")
   ```

---

## Expected Timeline

| Step | Duration (A4000 GPU) | Duration (CPU) |
|------|----------------------|----------------|
| YOLO Dataset Prep | 15-30 min | 30-60 min |
| YOLO Training (100 epochs) | 8-12 hours | Not recommended |
| Style Training (30 epochs × 3 models) | 2-4 hours | Not recommended |
| **Total** | **10-16 hours** | **Not feasible** |

**Note**: Training on CPU is not recommended due to extremely long training times (days to weeks).

---

## Model Checkpoints

All trained models are saved automatically:

```
phase2_outputs/
├── yolo_dataset/
│   ├── data.yaml
│   ├── category_mapping.json
│   └── images/ & labels/
├── yolo_training_runs/
│   └── finetune_294_classes/
│       ├── weights/
│       │   ├── best.pt          # Best YOLO model
│       │   └── last.pt          # Last checkpoint
│       ├── results.csv          # Training metrics
│       └── training_summary.json
├── best_efficientnet_style_classifier.pth
├── best_resnet_style_classifier.pth
├── best_vit_style_classifier.pth
├── improved_style_classifier_results.json
└── phase2_report_[timestamp].json
```

---

## Next Steps

After completing Phase 2:

1. **Deploy Models**
   - Export YOLO to ONNX for production
   - Create FastAPI endpoint for inference

2. **Evaluate on Test Set**
   - Collect new images not in training/validation
   - Measure real-world performance

3. **Iterate and Improve**
   - Analyze errors and edge cases
   - Collect more data for weak categories
   - Fine-tune ensemble weights based on validation

4. **Phase 3 Planning**
   - Real-time inference API
   - Mobile app integration
   - 3D room reconstruction
   - Style recommendation engine

---

## Support

For issues or questions:
- Check [Troubleshooting](#troubleshooting) section
- Review Phase 1 documentation for setup issues
- Check GPU availability: `nvidia-smi`
- Verify Python environment: `pip list | grep torch`

---

**Last Updated**: 2025-11-08
**Phase**: 2.0
**Status**: Production Ready
