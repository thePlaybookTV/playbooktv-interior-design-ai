# Paperspace R2 Training - Quick Start

**Training from fresh R2 images in 5 steps** 🚀

---

## Step 1: Update Code (30 seconds)

```bash
cd /notebooks/app
git pull origin main
```

---

## Step 2: Set Up R2 Credentials (2 minutes)

Create `.env` file with your R2 credentials:

```bash
cd /notebooks/app

cat > .env << 'EOF'
# Cloudflare R2
CLOUDFLARE_ACCOUNT_ID=your_account_id_here
R2_ACCESS_KEY_ID=your_access_key_here
R2_SECRET_ACCESS_KEY=your_secret_key_here
R2_BUCKET_NAME=your_bucket_name_here
EOF
```

**Replace the placeholder values with your actual R2 credentials!**

---

## Step 3: Test R2 Connection (30 seconds)

```bash
cd /notebooks/app

# Test connection
python src/data_collection/cloudflare_r2_downloader.py --test
```

**Expected output:**
```
✅ Connected to R2 bucket: your_bucket_name
✅ Connection test successful!
```

**If you see errors**: Check your credentials in `.env`

---

## Step 4A: Test with 100 Images First (Recommended - 2-3 hours)

```bash
cd /notebooks/app

# Test run with just 100 images
python scripts/run_r2_to_phase2_training.py \
    --db database_test.duckdb \
    --output ./test_outputs \
    --max-images 100 \
    --yolo-epochs 5 \
    --style-epochs 3
```

This will:
- Download 100 images from R2
- Process them with Phase 1
- Train for 5 YOLO epochs + 3 style epochs
- Let you verify everything works before the long 12-hour run

**Wait ~2-3 hours for this to complete.**

---

## Step 4B: Full Training (After Test Succeeds - 14-18 hours)

```bash
cd /notebooks/app

# Full training with all images
python scripts/run_r2_to_phase2_training.py \
    --db database_r2_full.duckdb \
    --output ./r2_phase2_outputs
```

This will:
- Download ALL images from R2
- Process with Phase 1 (YOLO + classification)
- Train YOLO for 100 epochs (~8-12 hours)
- Train 3 style classifiers for 30 epochs each (~3-5 hours)

**Go do something else for 14-18 hours!** ☕

---

## Step 5: Deploy to API (5 minutes)

Once training completes:

### 5.1 Update API Configuration

```bash
cd /notebooks/app
nano api/.env
```

Update these lines:
```bash
YOLO_MODEL_PATH=../r2_phase2_outputs/yolo_training_runs/finetune_294_classes/weights/best.pt
EFFICIENTNET_PATH=../r2_phase2_outputs/best_efficientnet_style_classifier.pth
RESNET_PATH=../r2_phase2_outputs/best_resnet_style_classifier.pth
VIT_PATH=../r2_phase2_outputs/best_vit_style_classifier.pth
DATABASE_PATH=../database_r2_full.duckdb
```

Save: `Ctrl+O`, `Enter`, `Ctrl+X`

### 5.2 Restart API

```bash
cd /notebooks/app

# Stop old API
pkill -f "uvicorn api.main:app"

# Start with new models
./start_api.sh
```

### 5.3 Test API

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

### 5.4 Get Public URL

```bash
pip install pyngrok

python << 'EOF'
from pyngrok import ngrok
import time

url = ngrok.connect(8000)
print("\n" + "="*60)
print("🌐 YOUR PUBLIC API URL:")
print(f"   {url}")
print("="*60)
print("\nUse this in Modomo app!")
print("\nPress Ctrl+C to stop")
print("="*60 + "\n")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nStopped")
EOF
```

**Copy the URL** and use it in your Modomo app!

---

## Monitoring Progress

### Watch Training in Real-Time

```bash
# In a new terminal tab
cd /notebooks/app
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

## Troubleshooting

### "ModuleNotFoundError: No module named 'boto3'"

```bash
pip install boto3 botocore
```

### "Connection failed"

Check your `.env` file has correct R2 credentials.

### "Out of memory"

Reduce batch size:
```bash
python scripts/run_r2_to_phase2_training.py \
    --db database.duckdb \
    --output ./outputs \
    --yolo-batch-size 8
```

### "Out of disk space"

Clean up old files:
```bash
rm -rf phase2_outputs/ test_outputs/
```

---

## Advanced Options

### Download Only (No Training)

```bash
python src/data_collection/cloudflare_r2_downloader.py \
    --output ./r2_images \
    --max-images 1000
```

### Skip Download (Use Existing Images)

```bash
python scripts/run_r2_to_phase2_training.py \
    --db database.duckdb \
    --output ./outputs \
    --skip-download
```

### Download from Specific Folder

```bash
python scripts/run_r2_to_phase2_training.py \
    --db database.duckdb \
    --output ./outputs \
    --r2-prefix "training_images/"
```

---

## Complete Copy-Paste Workflow

Just copy this entire block into Paperspace terminal:

```bash
# Navigate to project
cd /notebooks/app

# Update code
git pull origin main

# Test R2 connection (make sure you created .env first!)
python src/data_collection/cloudflare_r2_downloader.py --test

# Run test with 100 images
python scripts/run_r2_to_phase2_training.py \
    --db database_test.duckdb \
    --output ./test_outputs \
    --max-images 100 \
    --yolo-epochs 5 \
    --style-epochs 3

# If test succeeds, run full training (in tmux/screen)
python scripts/run_r2_to_phase2_training.py \
    --db database_r2_full.duckdb \
    --output ./r2_phase2_outputs
```

---

## What You Get

After training completes, you'll have:

1. **YOLO Model** (130-150MB)
   - Fine-tuned on 294 interior design categories
   - At: `r2_phase2_outputs/yolo_training_runs/finetune_294_classes/weights/best.pt`

2. **Style Classifiers** (3 models, ~450MB total)
   - EfficientNet-B0: `r2_phase2_outputs/best_efficientnet_style_classifier.pth`
   - ResNet50: `r2_phase2_outputs/best_resnet_style_classifier.pth`
   - ViT-B/16: `r2_phase2_outputs/best_vit_style_classifier.pth`

3. **Database** (~50-200MB)
   - All image metadata and detections
   - At: `database_r2_full.duckdb`

4. **Training Logs**
   - YOLO training metrics
   - Style classifier accuracies
   - All in `r2_phase2_outputs/`

---

## Expected Results

### Before (Phase 1):
- 14 COCO object classes
- ~54% style accuracy
- Basic detection

### After (Phase 2 with R2 Data):
- 294 specific interior categories
- ~70%+ style accuracy
- Precise furniture detection
- Ensemble classification

---

## Timeline

| Step | Time |
|------|------|
| Download test images (100) | 1-2 min |
| Process test images | 15-30 min |
| Test YOLO training | 30-60 min |
| Test style training | 15-30 min |
| **Test Total** | **~2-3 hours** |
| | |
| Download all images | 10 min - 2 hours |
| Process all images | 2-6 hours |
| Full YOLO training | 8-12 hours |
| Full style training | 3-5 hours |
| **Full Total** | **~14-25 hours** |

---

## Next Steps

1. ✅ **Now**: Update code and test R2 connection
2. ✅ **Today**: Run test with 100 images (2-3 hours)
3. ✅ **Tonight**: Start full training before bed (14-18 hours)
4. ✅ **Tomorrow**: Wake up to trained models!
5. ✅ **Deploy**: Update API and get public URL
6. ✅ **Use**: Connect Modomo app and test

---

**Ready to train from R2!** 🚀

For detailed documentation, see [R2_TRAINING_GUIDE.md](R2_TRAINING_GUIDE.md)
