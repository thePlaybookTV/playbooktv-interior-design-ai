# Paperspace Phase 2 Training - Quick Start

## 🎯 What You'll Train

**Phase 2 Training** includes:
1. **YOLO Fine-tuning** - 294 interior-specific categories (8-12 hours)
2. **Ensemble Style Classifier** - 70%+ accuracy (2-4 hours)

**Total Time**: 10-16 hours on Paperspace GPU

---

## 📋 Prerequisites

### 1. Paperspace Gradient Setup
- **Machine Type**: GPU instance (A4000, A5000, or better)
- **Storage**: 50GB+ free space
- **RAM**: 16GB+ recommended

### 2. Required Files
- **Database**: Your DuckDB file with processed images
- **Project Code**: This repository cloned to Paperspace

---

## 🚀 Quick Start (Copy & Paste)

### Step 1: Clone Repository (if not already done)

```bash
cd /notebooks
git clone https://github.com/YOUR_REPO/playbooktv-interior-design-ai.git app
cd app
```

### Step 2: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Verify YOLO installation
python -c "from ultralytics import YOLO; print('✅ YOLO ready')"
```

### Step 3: Upload Your Database

Upload your DuckDB file to Paperspace:
- Via Paperspace UI: Upload `metadata.duckdb` to `/notebooks/app/`
- Via CLI (if database is on your machine): Use Paperspace CLI to upload

Expected location: `/notebooks/app/interior_design_data_hybrid/processed/metadata.duckdb`

### Step 4: Launch Training

Use the convenient launch script:

```bash
# Make script executable
chmod +x start_training.sh

# Run full Phase 2 training
./start_training.sh
```

**Or run manually:**

```bash
python scripts/run_phase2_training.py \
    --db ./interior_design_data_hybrid/processed/metadata.duckdb \
    --output ./phase2_outputs \
    --yolo-epochs 100 \
    --style-epochs 30 \
    --batch-size 16
```

---

## 📊 What to Expect

### Console Output:

```
================================================================================
🚀 PHASE 2 TRAINING PIPELINE
================================================================================

Database: /notebooks/app/interior_design_data_hybrid/processed/metadata.duckdb
Output Directory: /notebooks/app/phase2_outputs
Timestamp: 20250116_143022
================================================================================

================================================================================
📦 STEP 1: PREPARING YOLO DATASET
================================================================================

Analyzing database...
Found 12,453 images with detections
Creating YOLO format dataset...
   Train images: 9,962
   Val images: 2,491
   Total classes: 294

✅ YOLO dataset preparation complete!

================================================================================
🎯 STEP 2: FINE-TUNING YOLO ON 294 CATEGORIES
================================================================================

Loading base model: yolov8m.pt
Starting training...
Epoch 1/100: 100%|████████████| 623/623 [12:34<00:00, train_loss=2.134]
...
```

### Training Progress:

**Phase 1 - YOLO Fine-tuning** (8-12 hours):
- Epoch 1-20: Loss drops rapidly
- Epoch 20-50: Gradual improvement
- Epoch 50-100: Fine-tuning convergence

**Phase 2 - Style Classifier** (2-4 hours):
- Trains 3 models in parallel (EfficientNet, ResNet50, ViT)
- Each model takes 30 epochs
- Automatic ensemble creation

---

## 📁 Output Files

After training completes, you'll find:

```
phase2_outputs/
├── yolo_dataset/
│   ├── data.yaml
│   ├── train/
│   └── val/
├── yolo_training_runs/
│   └── finetune_294_classes/
│       └── weights/
│           └── best.pt          ← **Upload this to Modal**
├── style_classifier_outputs/
│   ├── best_efficientnet_style_classifier.pth  ← **Upload to Modal**
│   ├── best_resnet_style_classifier.pth        ← **Upload to Modal**
│   ├── best_vit_style_classifier.pth           ← **Upload to Modal**
│   └── ensemble_metadata.json
└── training_report_TIMESTAMP.txt
```

---

## 📤 Upload Models to Modal

After training completes, upload the models to Modal Volume:

```bash
# 1. Upload YOLO model
modal volume put modomo-models \
  phase2_outputs/yolo_training_runs/finetune_294_classes/weights/best.pt \
  /yolo/v1/best.pt

# 2. Upload Ensemble Classifier models
modal volume put modomo-models \
  phase2_outputs/style_classifier_outputs/best_efficientnet_style_classifier.pth \
  /ensemble/v1/efficientnet.pth

modal volume put modomo-models \
  phase2_outputs/style_classifier_outputs/best_resnet_style_classifier.pth \
  /ensemble/v1/resnet50.pth

modal volume put modomo-models \
  phase2_outputs/style_classifier_outputs/best_vit_style_classifier.pth \
  /ensemble/v1/vit.pth

# 3. Verify uploads
modal volume ls modomo-models /yolo/v1/
modal volume ls modomo-models /ensemble/v1/
```

---

## 🎛️ Advanced Options

### Train YOLO Only

```bash
python scripts/run_phase2_training.py \
    --skip-style \
    --yolo-epochs 100 \
    --batch-size 16
```

### Train Style Classifier Only

```bash
python scripts/run_phase2_training.py \
    --skip-yolo \
    --style-epochs 30 \
    --batch-size 32
```

### Adjust Batch Size (if GPU memory issues)

```bash
# Reduce batch size for smaller GPUs
python scripts/run_phase2_training.py \
    --batch-size 8  # or even 4
```

### Resume Training

If training is interrupted:

```bash
# YOLO auto-resumes from last.pt
python scripts/run_phase2_training.py --resume

# Or manually specify checkpoint
python scripts/run_phase2_training.py \
    --yolo-checkpoint ./phase2_outputs/yolo_training_runs/finetune_294_classes/weights/last.pt
```

---

## 🔍 Monitor Training

### Option 1: Watch Console Output

Training progress is displayed in real-time with loss curves and metrics.

### Option 2: TensorBoard (if available)

```bash
# In a separate terminal
tensorboard --logdir phase2_outputs/yolo_training_runs
```

### Option 3: Check GPU Usage

```bash
# In a separate terminal
watch -n 1 nvidia-smi
```

Expected GPU utilization: 80-95%

---

## 💰 Cost Estimate

**Paperspace Gradient Pricing** (approximate):
- **A4000 GPU**: ~$0.76/hour × 12 hours = **~$9.12**
- **A5000 GPU**: ~$1.38/hour × 10 hours = **~$13.80**
- **A100 GPU**: ~$3.09/hour × 8 hours = **~$24.72** (faster)

**Recommended**: A4000 or A5000 for cost-effectiveness

---

## 🐛 Troubleshooting

### Issue: CUDA Out of Memory

**Solution**: Reduce batch size

```bash
python scripts/run_phase2_training.py --batch-size 8
```

### Issue: Database not found

**Error**: `FileNotFoundError: metadata.duckdb`

**Solution**: Check database path
```bash
ls -lh interior_design_data_hybrid/processed/metadata.duckdb
```

Upload if missing.

### Issue: ModuleNotFoundError

**Error**: `ModuleNotFoundError: No module named 'src'`

**Solution**: Ensure you're in project root
```bash
cd /notebooks/app
python scripts/run_phase2_training.py ...
```

### Issue: Insufficient data

**Error**: `Not enough images for training (minimum 5000 required)`

**Solution**: You need to run Phase 1 data collection first. See [docs/training/QUICKSTART.md](docs/training/QUICKSTART.md).

---

## 📊 Expected Performance

### YOLO Fine-tuning Results:

| Metric | Expected Value |
|--------|---------------|
| **mAP@0.5** | 0.65-0.75 |
| **mAP@0.5:0.95** | 0.45-0.55 |
| **Precision** | 0.70-0.80 |
| **Recall** | 0.60-0.70 |

### Style Classifier Results:

| Model | Expected Accuracy |
|-------|------------------|
| **EfficientNet-B0** | 68-70% |
| **ResNet50** | 65-67% |
| **ViT B/16** | 63-65% |
| **Ensemble** | **70-75%** |

Compare to Phase 1: 53.8% → **+16-21% improvement!**

---

## ✅ Validation

After training completes, the script automatically validates:

1. **YOLO Validation**:
   - Runs on validation set
   - Generates confusion matrix
   - Saves metrics to report

2. **Style Classifier Validation**:
   - Tests each model individually
   - Creates ensemble predictions
   - Compares to Phase 1 baseline

Check the training report:
```bash
cat phase2_outputs/training_report_*.txt
```

---

## 🎉 Success Checklist

After training, you should have:

- [x] `best.pt` YOLO model (~100MB)
- [x] `best_efficientnet_style_classifier.pth` (~17MB)
- [x] `best_resnet_style_classifier.pth` (~90MB)
- [x] `best_vit_style_classifier.pth` (~330MB)
- [x] Training report with metrics
- [x] Models uploaded to Modal Volume
- [x] Modal function redeployed (auto-detects new models)

**Next Step**: Test your upgraded Modal pipeline with real images!

---

## 📞 Need Help?

**Common Questions**:
- How long should training take? → 10-16 hours on A4000
- Can I stop and resume? → Yes, YOLO auto-resumes from `last.pt`
- What if accuracy is lower? → Check training report, may need more epochs or data
- How do I know it's working? → Watch loss decrease over epochs

**Resources**:
- Full guide: [docs/training/PHASE2_TRAINING_GUIDE.md](docs/training/PHASE2_TRAINING_GUIDE.md)
- Quick start: [docs/training/QUICKSTART.md](docs/training/QUICKSTART.md)
- Architecture: [docs/deployment/ARCHITECTURE.md](docs/deployment/ARCHITECTURE.md)

---

**🚀 Ready to train? Run `./start_training.sh` and let it cook!**
