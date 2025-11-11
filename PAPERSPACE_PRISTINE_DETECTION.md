# Running Pristine Detector (SAM2 + YOLO) on Paperspace

Complete guide for running furniture detection with YOLO bounding boxes and SAM2 segmentation masks on Paperspace GPU.

---

## 📋 Prerequisites

Before starting, upload these models to Paperspace Gradient:

### 1. YOLO Model
- **File**: `yolov8m.pt` (~50MB)
- **Download locally first**:
  ```bash
  wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt
  ```

### 2. SAM2 Checkpoint
- **File**: `sam2_hiera_large.pt` (~900MB)
- **Download locally first**:
  ```bash
  wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
  ```

### 3. Upload to Gradient Models
1. Go to **Gradient Console** → **Models**
2. Create a model called `detection-models`
3. Upload both files

---

## 🚀 Quick Start on Paperspace

### Step 1: Set Up Environment

```bash
cd /notebooks/app

# Link pre-uploaded models
mkdir -p checkpoints
ln -sf /models/detection-models/yolov8m.pt ./yolov8m.pt
ln -sf /models/detection-models/sam2_hiera_large.pt ./checkpoints/sam2_hiera_large.pt

# Verify models are accessible
ls -lh yolov8m.pt
ls -lh checkpoints/sam2_hiera_large.pt
```

**Expected output:**
```
lrwxr-xr-x  1 user  staff   50M  yolov8m.pt -> /models/detection-models/yolov8m.pt
lrwxr-xr-x  1 user  staff  900M  checkpoints/sam2_hiera_large.pt -> /models/detection-models/sam2_hiera_large.pt
```

### Step 2: Install SAM2

```bash
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

### Step 3: Run Detection on Your Images

**Option A: Using pristine_detector.py directly (standalone)**

```bash
cd /notebooks/app

# This will process all images in your database
python src/models/pristine_detector.py
```

**Option B: Using the batch processor (recommended for large datasets)**

```bash
cd /notebooks/app

python scripts/process_images_in_batches.py \
    --images /path/to/your/images \
    --db database_metadata.duckdb \
    --batch-size 50
```

---

## 📊 What Gets Detected

The pipeline detects these furniture items:
- Seating: couch, chair
- Bedroom: bed
- Dining: dining table
- Kitchen: refrigerator, oven, sink
- Electronics: tv, laptop
- Decor: potted plant, vase, clock
- Other: toilet, book

For each item detected, you get:
- **YOLO bounding box** (fast, ~30ms)
- **SAM2 segmentation mask** (precise, ~100ms)
- **Confidence scores**
- **Area percentages**
- **Mask quality scores**

---

## 💾 Database Schema

Detection results are saved to DuckDB with this schema:

### `furniture_detections` table:
```sql
CREATE TABLE furniture_detections (
    image_id VARCHAR,
    item_type VARCHAR,           -- e.g., 'couch', 'chair'
    confidence FLOAT,            -- YOLO confidence (0-1)
    bbox_x1 FLOAT,              -- Bounding box coordinates
    bbox_y1 FLOAT,
    bbox_x2 FLOAT,
    bbox_y2 FLOAT,
    area_percentage FLOAT,       -- % of image occupied
    mask_area INTEGER,           -- Pixels in SAM2 mask
    mask_score FLOAT,            -- SAM2 quality score
    has_mask BOOLEAN             -- Whether SAM2 succeeded
)
```

### `images` table gets updated:
```sql
ALTER TABLE images ADD COLUMN furniture_count INTEGER
```

---

## 🎮 Performance on Paperspace GPU

### A4000 GPU (recommended):
- **Speed**: ~3-4 images/second
- **Memory**: 16GB VRAM (plenty of headroom)
- **Time for 67K images**: ~5 hours
- **Auto-checkpointing**: Every 500 images

### A6000 GPU (faster):
- **Speed**: ~5-6 images/second
- **Time for 67K images**: ~3-4 hours

### CPU (not recommended):
- **Speed**: ~0.3 images/second
- **Time for 67K images**: ~60+ hours

---

## 🔄 Checkpointing & Resume

The processor automatically saves progress every 500 images:

**Checkpoint file**: `processing_checkpoint.json`

```json
{
  "processed": ["img1", "img2", ...],
  "timestamp": "2025-11-11T10:30:00",
  "total": 15000
}
```

**To resume after interruption:**
Just run the same command again - it will skip already processed images!

```bash
# Will automatically continue from last checkpoint
python src/models/pristine_detector.py
```

---

## 📈 Monitoring Progress

### In Real-Time:

```bash
# Watch the processing in terminal
python src/models/pristine_detector.py
```

You'll see:
```
📦 BATCH 1/134
   Processing images 1-500 of 67000
================================================================
Processing: 100%|████████████████| 500/500 [02:15<00:00, 3.7 images/s]

💾 Checkpoint: 500/67000 (293 min remaining)
```

### Check Database Stats While Running:

```bash
# In another terminal/notebook
python << 'EOF'
import duckdb
conn = duckdb.connect('interior_design_data_hybrid/processed/metadata.duckdb')

# How many images processed?
total = conn.execute("""
    SELECT COUNT(*)
    FROM images
    WHERE furniture_count IS NOT NULL
""").fetchone()[0]

print(f"📷 Images processed: {total:,}")

# What's been detected?
items = conn.execute("""
    SELECT item_type, COUNT(*) as count
    FROM furniture_detections
    GROUP BY item_type
    ORDER BY count DESC
    LIMIT 10
""").df()

print("\n🏆 Top 10 Items:")
print(items)
EOF
```

---

## 🧪 Testing First

Before processing all 67K images, test on a small subset:

```bash
cd /notebooks/app

python << 'EOF'
from pathlib import Path
import sys
sys.path.insert(0, '/notebooks/app')

from src.models.pristine_detector import PristineDetector

# Initialize detector
detector = PristineDetector()

# Test on one image
test_image = "/path/to/test/image.jpg"
result = detector.detect_with_masks(test_image)

print(f"\n✅ Detection test successful!")
print(f"   Items found: {result['count']}")
for item in result['items']:
    print(f"   - {item['type']}: {item['confidence']:.2f} confidence")
    print(f"     Has mask: {item['has_mask']}")
EOF
```

---

## 🔧 Troubleshooting

### "No module named 'sam2'" error

```bash
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

### "FileNotFoundError: checkpoints/sam2_hiera_large.pt"

```bash
# Check if symlink exists
ls -la checkpoints/sam2_hiera_large.pt

# If not, recreate it
mkdir -p checkpoints
ln -sf /models/detection-models/sam2_hiera_large.pt ./checkpoints/sam2_hiera_large.pt
```

### "YOLO model not found"

```bash
# Check if symlink exists
ls -la yolov8m.pt

# If not, recreate it
ln -sf /models/detection-models/yolov8m.pt ./yolov8m.pt
```

### Out of GPU memory

```bash
# Reduce batch size in the code or use smaller YOLO model
# Edit pristine_detector.py and change:
# self.yolo = YOLO('yolov8s.pt')  # Smaller model
```

### Processing is slow (using CPU instead of GPU)

```bash
# Check CUDA availability
python << 'EOF'
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
EOF
```

If CUDA is not available, make sure you're using a GPU machine type in Paperspace.

---

## 📊 After Processing: View Statistics

```bash
cd /notebooks/app

python << 'EOF'
import duckdb
import sys

db_path = "interior_design_data_hybrid/processed/metadata.duckdb"
conn = duckdb.connect(db_path)

print("="*70)
print("📊 DETECTION STATISTICS")
print("="*70)

# Overall stats
total = conn.execute("""
    SELECT COUNT(*)
    FROM images
    WHERE furniture_count IS NOT NULL
""").fetchone()[0]

stats = conn.execute("""
    SELECT
        AVG(furniture_count) as avg_count,
        MAX(furniture_count) as max_count,
        MIN(furniture_count) as min_count
    FROM images
    WHERE furniture_count IS NOT NULL
""").fetchone()

print(f"\n📷 Images processed: {total:,}")
print(f"🪑 Average items per image: {stats[0]:.1f}")
print(f"📊 Max items in one image: {stats[1]}")
print(f"📊 Min items in one image: {stats[2]}")

# Items with masks
with_masks = conn.execute("""
    SELECT COUNT(*)
    FROM furniture_detections
    WHERE has_mask = TRUE
""").fetchone()[0]

total_detections = conn.execute("""
    SELECT COUNT(*)
    FROM furniture_detections
""").fetchone()[0]

print(f"\n🎭 Detections with SAM2 masks: {with_masks:,}/{total_detections:,} ({with_masks/total_detections*100:.1f}%)")

# Most common items
print("\n🏆 Top 10 Detected Items:")
items = conn.execute("""
    SELECT item_type, COUNT(*) as count
    FROM furniture_detections
    GROUP BY item_type
    ORDER BY count DESC
    LIMIT 10
""").df()

for _, row in items.iterrows():
    print(f"   {row['item_type']:20s}: {row['count']:6,}")

conn.close()
EOF
```

---

## 🔗 Integration with Phase 2 Training

After detection completes, use the results for training:

```bash
cd /notebooks/app

# Run Phase 2 training with detected furniture
python scripts/run_phase2_training.py \
    --db interior_design_data_hybrid/processed/metadata.duckdb \
    --output ./phase2_outputs \
    --skip-download  # Models already detected
```

---

## ⏱️ Complete Workflow Timeline

| Step | Time | Command |
|------|------|---------|
| Upload models to Gradient | 10 min | Manual upload |
| Link models in workspace | 30s | `ln -sf` commands |
| Install SAM2 | 2 min | `pip install` |
| Test on 1 image | 10s | Test script |
| Process 67K images | 5 hours | `python pristine_detector.py` |
| View statistics | 10s | Stats script |
| **Total** | **~5-6 hours** | |

---

## 💡 Pro Tips

1. **Use A4000 GPU**: Perfect balance of speed and cost
2. **Start with test image**: Verify models load correctly
3. **Monitor first 100 images**: Make sure checkpointing works
4. **Check stats periodically**: Ensure detections look reasonable
5. **Don't worry about interruptions**: Checkpointing handles them
6. **Keep terminal open**: Or use `tmux`/`screen` to run in background

---

## 🎉 Expected Results

After processing all images, you should see:

```
✅ PROCESSING COMPLETE

📊 Summary:
   Total images: 67,000
   Successfully processed: 67,000
   Skipped (already in DB): 0
   Failed: 0

📷 Images processed: 67,000
🪑 Average items per image: 8.3
🎭 Detections with SAM2 masks: 556,100/556,100 (100%)

🏆 Top 10 Detected Items:
   chair               : 89,234
   couch               : 67,891
   potted plant        : 45,678
   dining table        : 34,567
   bed                 : 23,456
   tv                  : 21,234
   vase                : 19,876
   book                : 15,432
   sink                : 12,345
   laptop              : 9,876
```

---

## 🆘 Need Help?

If you encounter issues:

1. Check this guide's troubleshooting section
2. Verify GPU is active: `nvidia-smi`
3. Check model paths: `ls -lh yolov8m.pt checkpoints/`
4. Review error messages in terminal
5. Check checkpoint file exists: `ls -lh processing_checkpoint.json`

---

**Ready to detect! 🚀**

The pristine detector will give you high-quality furniture detections with both fast YOLO bounding boxes and precise SAM2 segmentation masks - perfect for training Phase 2 models.
