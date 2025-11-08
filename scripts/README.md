# Scripts Directory

This directory contains executable scripts for training and running the PlaybookTV Interior Design AI system.

---

## Available Scripts

### run_phase2_training.py

**Purpose**: Complete Phase 2 training pipeline for YOLO fine-tuning and improved style classification.

**Usage**:

```bash
# Full pipeline (YOLO + Style Classification)
python scripts/run_phase2_training.py \
    --db ./interior_design_data_hybrid/processed/metadata.duckdb \
    --output ./phase2_outputs \
    --yolo-epochs 100 \
    --style-epochs 30 \
    --batch-size 16

# YOLO training only
python scripts/run_phase2_training.py \
    --skip-style \
    --yolo-epochs 100

# Style classification only
python scripts/run_phase2_training.py \
    --skip-yolo \
    --style-epochs 30
```

**Parameters**:
- `--db`: Path to DuckDB database (required)
- `--output`: Output directory for results (default: `./phase2_outputs`)
- `--skip-yolo`: Skip YOLO training
- `--skip-style`: Skip style classification training
- `--yolo-epochs`: Number of YOLO training epochs (default: 100)
- `--style-epochs`: Number of style classifier epochs (default: 30)
- `--batch-size`: Batch size for training (default: 16)

**Output**:
- YOLO dataset in YOLO format
- Fine-tuned YOLO model (294 classes)
- Three style classification models (EfficientNet, ResNet50, ViT)
- Comprehensive training report

**Time**: 10-16 hours on A4000 GPU

---

## Quick Commands

### Test Your Setup

```bash
# Check if database exists
python -c "import duckdb; conn = duckdb.connect('./interior_design_data_hybrid/processed/metadata.duckdb'); print(f'Images: {conn.execute(\"SELECT COUNT(*) FROM images\").fetchone()[0]}')"

# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Verify dependencies
python -c "import ultralytics; import torchvision; import efficientnet_pytorch; print('All dependencies installed!')"
```

### Quick Training Tests

```bash
# Quick test with fewer epochs (for validation)
python scripts/run_phase2_training.py \
    --db ./interior_design_data_hybrid/processed/metadata.duckdb \
    --yolo-epochs 5 \
    --style-epochs 2 \
    --batch-size 8

# YOLO only (faster iteration)
python scripts/run_phase2_training.py \
    --skip-style \
    --yolo-epochs 10 \
    --batch-size 16
```

---

## Common Issues

### CUDA Out of Memory

**Solution**: Reduce batch size
```bash
python scripts/run_phase2_training.py --batch-size 8
```

### Database Not Found

**Solution**: Check database path
```bash
# Find database
find . -name "*.duckdb"

# Use correct path
python scripts/run_phase2_training.py --db /correct/path/to/metadata.duckdb
```

### Missing Dependencies

**Solution**: Install required packages
```bash
pip install ultralytics efficientnet-pytorch timm
```

---

## Advanced Usage

### Resume Training

YOLO training can be resumed automatically from the last checkpoint:

```python
from src.models.yolo_finetune import YOLOFineTuner

finetuner = YOLOFineTuner(data_yaml="./yolo_dataset/data.yaml")
finetuner.train(resume=True)  # Resumes from last checkpoint
```

### Custom Configuration

```python
from scripts.run_phase2_training import Phase2Pipeline

# Create custom pipeline
pipeline = Phase2Pipeline(
    db_path="./metadata.duckdb",
    output_dir="./custom_output"
)

# Run specific steps
data_yaml, result = pipeline.step1_prepare_yolo_dataset()
train_results, val_results = pipeline.step2_finetune_yolo(
    data_yaml=data_yaml,
    epochs=50,  # Custom epochs
    batch_size=24  # Custom batch size
)
```

---

## Monitoring Training

### YOLO Training

Training progress is displayed in real-time:
```
Epoch   GPU_mem   box_loss   cls_loss   dfl_loss   Instances   Size
1/100   7.5G      1.234      0.567      0.891      256         640
```

Results are saved to:
- `yolo_training_runs/finetune_294_classes/results.csv`
- `yolo_training_runs/finetune_294_classes/results.png`

### Style Classification

Per-model training progress:
```
Training efficientnet...
Epoch 1/30: Train Acc=0.4567, Val Acc=0.4321
Epoch 2/30: Train Acc=0.5234, Val Acc=0.4987
...
💾 Saved best efficientnet model (Val Acc: 0.6789)
```

---

## Performance Optimization

### For Faster Training

1. **Use smaller YOLO model**:
   ```python
   # In run_phase2_training.py, modify:
   model_size='yolov8s.pt'  # Instead of yolov8m.pt
   ```

2. **Reduce image size**:
   ```python
   image_size=512  # Instead of 640
   ```

3. **Use mixed precision** (already enabled):
   - Automatic Mixed Precision (AMP) is on by default
   - Reduces memory usage and speeds up training

### For Better Accuracy

1. **Increase epochs**:
   ```bash
   --yolo-epochs 150 --style-epochs 50
   ```

2. **Larger batch size** (if GPU allows):
   ```bash
   --batch-size 32
   ```

3. **Use larger YOLO model**:
   ```python
   model_size='yolov8l.pt'  # or yolov8x.pt
   ```

---

## Workflow Examples

### Development Workflow

1. **Test setup** (5 min):
   ```bash
   python scripts/run_phase2_training.py --yolo-epochs 1 --style-epochs 1 --batch-size 4
   ```

2. **Quick iteration** (2-3 hours):
   ```bash
   python scripts/run_phase2_training.py --yolo-epochs 20 --style-epochs 10
   ```

3. **Full training** (10-16 hours):
   ```bash
   python scripts/run_phase2_training.py --yolo-epochs 100 --style-epochs 30
   ```

### Production Workflow

1. **Prepare dataset**:
   ```bash
   python -c "from src.models.yolo_dataset_prep import YOLODatasetBuilder; builder = YOLODatasetBuilder('db.duckdb', 'yolo_dataset'); builder.prepare_dataset()"
   ```

2. **Train YOLO** (8-12 hours):
   ```bash
   python scripts/run_phase2_training.py --skip-style --yolo-epochs 100
   ```

3. **Validate YOLO**:
   ```bash
   python src/models/yolo_finetune.py --mode validate --weights ./phase2_outputs/yolo_training_runs/finetune_294_classes/weights/best.pt
   ```

4. **Train Style** (2-4 hours):
   ```bash
   python scripts/run_phase2_training.py --skip-yolo --style-epochs 30
   ```

5. **Evaluate ensemble**:
   ```python
   # Results automatically saved to improved_style_classifier_results.json
   ```

6. **Generate report**:
   ```bash
   # Report automatically generated in phase2_outputs/
   ```

---

## Documentation

For more information, see:
- [PHASE2_GUIDE.md](../docs/PHASE2_GUIDE.md) - Comprehensive guide
- [PHASE2_QUICKSTART.md](../PHASE2_QUICKSTART.md) - Quick reference
- [PHASE2_SUMMARY.md](../PHASE2_SUMMARY.md) - Implementation summary
- [PHASE1_VS_PHASE2.md](../docs/PHASE1_VS_PHASE2.md) - Detailed comparison

---

## Contributing

When adding new scripts:

1. Add executable permissions:
   ```bash
   chmod +x scripts/new_script.py
   ```

2. Include shebang:
   ```python
   #!/usr/bin/env python3
   ```

3. Add docstring:
   ```python
   """
   Script Name and Purpose

   Usage:
       python scripts/new_script.py --arg value
   """
   ```

4. Update this README

---

## License

Same as main project.

---

**Last Updated**: 2025-11-08
