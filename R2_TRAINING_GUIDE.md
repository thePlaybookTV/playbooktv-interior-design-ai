# R2-to-Phase2 Training Pipeline

Complete pipeline to train models from scratch using fresh images in Cloudflare R2.

## What This Does

```
Fresh R2 Images
    ↓
1. Download from Cloudflare R2
    ↓
2. Phase 1 Processing (YOLO detection + classification)
    ↓
3. Store detections in DuckDB
    ↓
4. Prepare YOLO dataset (294 categories)
    ↓
5. Train Phase 2 YOLO model
    ↓
6. Train ensemble style classifiers
    ↓
Ready for API deployment!
```

---

## Prerequisites

### 1. Set Up Environment Variables

Create a `.env` file in `/notebooks/app/` with your R2 credentials:

```bash
# Cloudflare R2 Credentials
CLOUDFLARE_ACCOUNT_ID=your_account_id
R2_ACCESS_KEY_ID=your_access_key_id
R2_SECRET_ACCESS_KEY=your_secret_access_key
R2_BUCKET_NAME=your_bucket_name

# Other API keys (if needed for Phase 1 processing)
OPENAI_API_KEY=your_openai_key  # For embeddings (optional)
```

### 2. Verify R2 Connection

```bash
cd /notebooks/app

# Test R2 connection
python src/data_collection/cloudflare_r2_downloader.py --test
```

Should see: `✅ Connection test successful!`

### 3. List Images in R2

```bash
# See what's in your R2 bucket
python src/data_collection/cloudflare_r2_downloader.py --list
```

---

## Running the Pipeline

### Full Pipeline (One Command)

```bash
cd /notebooks/app

# Run everything - download, process, train
python scripts/run_r2_to_phase2_training.py \
    --db database_r2_training.duckdb \
    --output ./r2_phase2_outputs
```

**This will take ~12-16 hours** depending on:
- Number of images
- GPU speed
- Internet speed for download

---

### Step-by-Step (Recommended for Testing)

#### Step 1: Download from R2 (Test with 100 images first)

```bash
cd /notebooks/app

# Download only 100 images for testing
python scripts/run_r2_to_phase2_training.py \
    --db database_r2_test.duckdb \
    --output ./r2_test_outputs \
    --max-images 100 \
    --yolo-epochs 5 \
    --style-epochs 3
```

#### Step 2: Once Test Works, Run Full Training

```bash
cd /notebooks/app

# Run with all images
python scripts/run_r2_to_phase2_training.py \
    --db database_r2_full.duckdb \
    --output ./r2_phase2_outputs
```

---

## Advanced Options

### Download Only (No Training)

```bash
# Just download images from R2
python src/data_collection/cloudflare_r2_downloader.py \
    --output ./r2_images \
    --max-images 1000
```

### Skip Download (Use Existing Images)

If you already downloaded images:

```bash
python scripts/run_r2_to_phase2_training.py \
    --db database.duckdb \
    --output ./outputs \
    --skip-download
```

### Skip Phase 1 (Use Existing Database)

If you already processed images and have a database:

```bash
python scripts/run_r2_to_phase2_training.py \
    --db existing_database.duckdb \
    --output ./outputs \
    --skip-download \
    --skip-phase1
```

### Download from Specific R2 Folder

```bash
python scripts/run_r2_to_phase2_training.py \
    --db database.duckdb \
    --r2-prefix "interior_images/" \
    --output ./outputs
```

### Quick Training (Fewer Epochs for Testing)

```bash
python scripts/run_r2_to_phase2_training.py \
    --db database.duckdb \
    --output ./outputs \
    --yolo-epochs 10 \
    --style-epochs 5
```

---

## What Each Step Does

### Step 1: Download from R2 (10 min - 2 hours)
- Downloads images from Cloudflare R2
- Saves to `{output_dir}/r2_images/`
- Skips already downloaded images

### Step 2: Phase 1 Processing (2-6 hours)
- Runs YOLO object detection on each image
- Classifies room type (living room, bedroom, etc.)
- Classifies design style (modern, rustic, etc.)
- Stores all detections in DuckDB
- **Requires:** GPU for speed (works on CPU but slower)

### Step 3: Prepare YOLO Dataset (2-5 min)
- Converts DuckDB detections to YOLO format
- Creates train/val splits (80/20 by default)
- Maps COCO classes to 294-category taxonomy
- Copies images to YOLO training directories

### Step 4: Fine-tune YOLO (8-12 hours)
- Fine-tunes YOLOv8 on 294 interior design categories
- 100 epochs by default
- Saves best model to `{output_dir}/yolo_training_runs/finetune_294_classes/weights/best.pt`

### Step 5: Train Style Classifiers (3-5 hours)
- Trains EfficientNet-B0
- Trains ResNet50
- Trains ViT-B/16
- Creates ensemble classifier
- Saves models to `{output_dir}/best_*_style_classifier.pth`

---

## Monitoring Progress

### Watch Training Logs

```bash
# In a separate terminal
tail -f r2_phase2_outputs/yolo_training_runs/training.log
```

### Check GPU Usage

```bash
watch -n 1 nvidia-smi
```

### Check Disk Space

```bash
df -h /notebooks
```

---

## After Training Completes

### 1. Verify Models Exist

```bash
cd /notebooks/app

# Check YOLO model
ls -lh r2_phase2_outputs/yolo_training_runs/finetune_294_classes/weights/best.pt

# Check style classifiers
ls -lh r2_phase2_outputs/best_*_style_classifier.pth

# Check database
ls -lh database_r2_full.duckdb
```

### 2. Update API Configuration

Edit `api/.env`:

```bash
# Update model paths
YOLO_MODEL_PATH=../r2_phase2_outputs/yolo_training_runs/finetune_294_classes/weights/best.pt
EFFICIENTNET_PATH=../r2_phase2_outputs/best_efficientnet_style_classifier.pth
RESNET_PATH=../r2_phase2_outputs/best_resnet_style_classifier.pth
VIT_PATH=../r2_phase2_outputs/best_vit_style_classifier.pth
DATABASE_PATH=../database_r2_full.duckdb
```

### 3. Restart API

```bash
cd /notebooks/app

# Stop old API
pkill -f "uvicorn api.main:app"

# Start with new models
./start_api.sh
```

### 4. Test API

```bash
curl http://localhost:8000/models/info
```

Should return:
```json
{
  "yolo": {
    "loaded": true,
    "classes": 294
  },
  "style_ensemble": {
    "loaded": true,
    "models": ["efficientnet", "resnet50", "vit"]
  }
}
```

---

## Troubleshooting

### "No images downloaded from R2"

**Check credentials:**
```bash
python << 'EOF'
import os
from dotenv import load_dotenv
load_dotenv()

print("Checking .env variables:")
print(f"  CLOUDFLARE_ACCOUNT_ID: {'✅' if os.getenv('CLOUDFLARE_ACCOUNT_ID') else '❌'}")
print(f"  R2_ACCESS_KEY_ID: {'✅' if os.getenv('R2_ACCESS_KEY_ID') else '❌'}")
print(f"  R2_SECRET_ACCESS_KEY: {'✅' if os.getenv('R2_SECRET_ACCESS_KEY') else '❌'}")
print(f"  R2_BUCKET_NAME: {'✅' if os.getenv('R2_BUCKET_NAME') else '❌'}")
EOF
```

**Test connection:**
```bash
python src/data_collection/cloudflare_r2_downloader.py --test
```

### "No images were processed in Phase 1"

This means YOLO couldn't load or process the images. Check:

```bash
# Verify images exist
ls -lh r2_phase2_outputs/r2_images/ | head -10

# Check if they're valid images
file r2_phase2_outputs/r2_images/*.jpg | head -5
```

### "No training images in YOLO dataset"

This means no detections were mapped to the 294 categories. Check:

```bash
# See what's in the database
python << 'EOF'
import duckdb
conn = duckdb.connect('database_r2_full.duckdb')
classes = conn.execute("""
    SELECT item_type, COUNT(*) as count
    FROM furniture_detections
    GROUP BY item_type
    ORDER BY count DESC
""").df()
print(classes)
conn.close()
EOF
```

If you see classes not in the mapping, update `src/models/yolo_dataset_prep.py` line 43-93.

### Out of Memory

**Reduce batch size:**
```bash
python scripts/run_r2_to_phase2_training.py \
    --db database.duckdb \
    --output ./outputs \
    --yolo-batch-size 8  # Default is 16
```

### Out of Disk Space

**Check space:**
```bash
df -h /notebooks
```

**Clean up old runs:**
```bash
rm -rf phase2_outputs/
rm -rf r2_test_outputs/
```

---

## Complete Example Workflow

```bash
# 1. Navigate to project
cd /notebooks/app

# 2. Pull latest code
git pull origin main

# 3. Verify R2 connection
python src/data_collection/cloudflare_r2_downloader.py --test

# 4. List images in bucket
python src/data_collection/cloudflare_r2_downloader.py --list | head -20

# 5. Test with 100 images first
python scripts/run_r2_to_phase2_training.py \
    --db database_test.duckdb \
    --output ./test_outputs \
    --max-images 100 \
    --yolo-epochs 5 \
    --style-epochs 3

# 6. If test succeeds, run full training
python scripts/run_r2_to_phase2_training.py \
    --db database_r2_full.duckdb \
    --output ./r2_phase2_outputs

# 7. Wait 12-16 hours...

# 8. Update API config
nano api/.env  # Update model paths

# 9. Restart API
./start_api.sh

# 10. Test it works
curl http://localhost:8000/models/info

# 11. Get public URL
python << 'EOF'
from pyngrok import ngrok
import time
url = ngrok.connect(8000)
print(f"\n🌐 Public URL: {url}\n")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nStopped")
EOF
```

---

## Expected Timeline

| Step | Time | What's Happening |
|------|------|------------------|
| R2 Download | 10 min - 2 hrs | Downloading images (depends on count & speed) |
| Phase 1 Processing | 2-6 hours | YOLO detection + classification |
| YOLO Dataset Prep | 2-5 min | Converting to YOLO format |
| YOLO Training | 8-12 hours | Fine-tuning on 294 categories |
| Style Classifier Training | 3-5 hours | Training 3 models + ensemble |
| **Total** | **~14-25 hours** | Full pipeline |

---

## Tips for Success

1. **Start Small**: Test with `--max-images 100` first
2. **Monitor GPU**: Use `nvidia-smi` to ensure GPU is being used
3. **Check Disk Space**: Phase 2 training uses ~50-100GB
4. **Use tmux/screen**: So training continues if you disconnect
5. **Save Outputs**: Keep both test and full outputs for comparison

---

## Questions?

- Check logs in `{output_dir}/`
- Look at database with DuckDB CLI: `duckdb database_r2_full.duckdb`
- Test individual components separately
- Verify each step completes before moving to next

---

**You're ready to train from R2!** 🚀
