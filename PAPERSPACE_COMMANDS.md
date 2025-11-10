# Paperspace Quick Commands

Copy and paste these commands into your Paperspace terminal.

---

## 🔄 Update Code with Fixes

```bash
cd /notebooks/app
git pull origin main
```

**Expected output:**
```
Updating 5199cd6..6e0627f
Fast-forward
 FIX_DATASET_PREP.md              | 303 ++++++++++++++++++++++
 api/main_simple.py               | 213 +++++++++++++++
 src/models/yolo_dataset_prep.py  |  54 +++-
 3 files changed, 522 insertions(+), 5 deletions(-)
```

---

## 🧪 Test the Fix (2 minutes)

```bash
cd /notebooks/app

# Quick test of dataset preparation
python << 'EOF'
import sys
sys.path.insert(0, '/notebooks/app')

from src.models.yolo_dataset_prep import YOLODatasetBuilder

print("🔧 Testing YOLO dataset preparation...\n")

builder = YOLODatasetBuilder(
    db_path='database_metadata.duckdb',
    output_dir='phase2_outputs/yolo_dataset_test'
)

stats = builder.prepare_dataset(train_split=0.8, min_confidence=0.5)

print("\n" + "="*60)
print("✅ TEST SUCCESSFUL!")
print(f"   Train images: {stats['train_images']}")
print(f"   Val images: {stats['val_images']}")
print(f"   Classes: {stats['num_classes']}")
print("="*60)
print("\nIf you see non-zero image counts above, the fix worked! 🎉")
print("You can now run the full training.\n")
EOF
```

---

## 🚀 Run Full Training (12 hours)

**Once the test above shows images being copied:**

```bash
cd /notebooks/app

# Stop any running processes first
# (Press Ctrl+C if something is running)

# Start Phase 2 training
python scripts/run_phase2_training.py \
    --db database_metadata.duckdb \
    --output ./phase2_outputs
```

**Monitor progress:**
The training will show:
- Dataset prep progress (3-5 mins)
- YOLO training progress (8 hours, 100 epochs)
- Style classifier training (4 hours, 30 epochs × 3 models)

You can close the terminal and come back - the process will keep running.

---

## 📊 Check Progress (While Training)

```bash
# See YOLO training progress
tail -n 50 phase2_outputs/yolo_training_runs/training.log

# Or watch in real-time
tail -f phase2_outputs/yolo_training_runs/training.log

# Check if models are being created
ls -lh phase2_outputs/yolo_training_runs/finetune_294_classes/weights/
```

---

## ✅ After Training Completes

**Verify models exist:**
```bash
cd /notebooks/app

# Check YOLO model
ls -lh phase2_outputs/yolo_training_runs/finetune_294_classes/weights/best.pt

# Check style classifiers
ls -lh best_*_style_classifier.pth
```

**Should see:**
```
-rw-r--r-- 1 user user 130M  best.pt
-rw-r--r-- 1 user user  17M  best_efficientnet_style_classifier.pth
-rw-r--r-- 1 user user  94M  best_resnet_style_classifier.pth
-rw-r--r-- 1 user user 327M  best_vit_style_classifier.pth
```

---

## 🔧 Update API with New Models

```bash
cd /notebooks/app

# Edit .env to point to new models
nano api/.env
```

**Update these lines:**
```bash
YOLO_MODEL_PATH=../phase2_outputs/yolo_training_runs/finetune_294_classes/weights/best.pt
EFFICIENTNET_PATH=../best_efficientnet_style_classifier.pth
RESNET_PATH=../best_resnet_style_classifier.pth
VIT_PATH=../best_vit_style_classifier.pth
```

Save with `Ctrl+O`, `Enter`, then exit with `Ctrl+X`.

---

## 🌐 Restart API with New Models

```bash
cd /notebooks/app

# Stop old API (Ctrl+C if running in terminal)
# Or find and kill:
pkill -f "uvicorn api.main:app"

# Start with new models
./start_api.sh
```

**Should see:**
```
🤖 Checking for models...
  ✅ Phase 2 YOLO model found
  ✅ EfficientNet model found
  ✅ ResNet model found
  ✅ ViT model found

🚀 Starting API Server
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

## 🧪 Test API with New Models

```bash
# In another terminal, or in Paperspace notebook:

curl http://localhost:8000/models/info
```

**Should return:**
```json
{
  "yolo": {
    "loaded": true,
    "status": "ready",
    "classes": 294
  },
  "style_ensemble": {
    "loaded": true,
    "status": "ready",
    "models": ["efficientnet", "resnet50", "vit"]
  }
}
```

---

## 🌐 Get Public URL (For Modomo)

**In a NEW Paperspace terminal** (keep API running in first one):

```bash
pip install pyngrok

python << 'EOF'
from pyngrok import ngrok
import time

public_url = ngrok.connect(8000)
print("\n" + "="*60)
print("🌐 YOUR PUBLIC API URL:")
print(f"   {public_url}")
print("="*60)
print("\nUse this in your Modomo app!")
print("\nAPI Endpoints:")
print(f"  • Health:  {public_url}/health")
print(f"  • Analyze: {public_url}/analyze")
print(f"  • Docs:    {public_url}/docs")
print("\nPress Ctrl+C to stop tunnel")
print("="*60 + "\n")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nTunnel stopped")
EOF
```

**Copy the URL** (like `https://abc123.ngrok.io`) and use it in your Modomo app!

---

## 🆘 Troubleshooting

### Training fails again?

```bash
# Check what classes are in database
python << 'EOF'
import duckdb
conn = duckdb.connect('database_metadata.duckdb')
result = conn.execute("""
    SELECT item_type, COUNT(*) as count
    FROM furniture_detections
    GROUP BY item_type
    ORDER BY count DESC
""").df()
print(result)
EOF
```

If you see classes not in the mapping, let me know and I'll add them.

### Out of memory during training?

```bash
# Reduce batch size
python scripts/run_phase2_training.py \
    --db database_metadata.duckdb \
    --output ./phase2_outputs \
    --yolo-batch-size 8  # Default is 16
```

### Need to start over?

```bash
# Clean up previous attempts
rm -rf phase2_outputs/
mkdir -p phase2_outputs

# Start fresh
python scripts/run_phase2_training.py \
    --db database_metadata.duckdb \
    --output ./phase2_outputs
```

---

## 📝 Complete Workflow Summary

1. **Update code**: `git pull origin main`
2. **Test fix**: Run the test script above (2 mins)
3. **Start training**: Run full training command (12 hours)
4. **Check progress**: Use tail commands while training
5. **Verify models**: Check file sizes after training
6. **Update API config**: Edit .env with new model paths
7. **Restart API**: `./start_api.sh`
8. **Get public URL**: Run ngrok script
9. **Connect Modomo**: Use the URL in your app
10. **Test it works**: Upload an image from Modomo!

---

## 📈 Expected Results

### Before (Phase 1):
- **Detection**: 14 COCO classes
- **Style Accuracy**: ~54%
- **Speed**: Fast but less accurate

### After (Phase 2):
- **Detection**: 294 specific interior design categories
- **Style Accuracy**: ~70%+
- **Speed**: Still fast (~200-300ms per image on GPU)

---

## ✨ Quick Reference

| Command | Purpose | Time |
|---------|---------|------|
| `git pull` | Update code | 10s |
| Test script | Verify fix works | 2-3 min |
| Full training | Train all models | ~12 hours |
| `./start_api.sh` | Start API server | 30s |
| ngrok script | Get public URL | 10s |

---

**You're all set!** Just follow the commands in order and you'll have a production-ready API with trained models. 🚀

Need help at any step? Check [FIX_DATASET_PREP.md](FIX_DATASET_PREP.md) for detailed troubleshooting.
