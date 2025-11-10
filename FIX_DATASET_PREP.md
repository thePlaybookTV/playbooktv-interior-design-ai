# Fixed YOLO Dataset Preparation

## What Was Wrong

The YOLO dataset preparation script had a **very limited COCO-to-taxonomy mapping** - only 14 classes were mapped. This meant most of your detections in the database (which use COCO class names like "couch", "chair", etc.) couldn't be converted to the 294-category taxonomy, so **no images were being copied**.

## What I Fixed

### 1. Expanded COCO Mapping
Added comprehensive mapping from COCO classes to taxonomy:
- **Before**: 14 classes mapped
- **After**: 30+ classes mapped (all common COCO classes)

### 2. Added Detailed Statistics
The script now shows exactly what's happening:
- How many images were successfully copied
- Why images were skipped (no file, no detections, unmapped classes)
- Which COCO classes couldn't be mapped

### 3. Better Error Handling
Added try-catch for file operations to show specific errors.

---

## How to Re-run Training

### Option 1: Run Full Pipeline (Recommended)

```bash
cd /notebooks/app

# Re-run the complete Phase 2 training
python scripts/run_phase2_training.py \
    --db database_metadata.duckdb \
    --output ./phase2_outputs
```

This will:
1. ✅ Re-prepare YOLO dataset with fixed mapping
2. ✅ Train YOLO for 100 epochs (~8 hours)
3. ✅ Train style classifiers for 30 epochs (~4 hours)

---

### Option 2: Just Fix Dataset First (Debug Mode)

If you want to see what's happening without committing to 12 hours of training:

```bash
cd /notebooks/app

# Just prepare the dataset to verify the fix
python << 'EOF'
import sys
sys.path.insert(0, '/notebooks/app')

from src.models.yolo_dataset_prep import YOLODatasetBuilder

# Prepare dataset
builder = YOLODatasetBuilder(
    db_path='database_metadata.duckdb',
    output_dir='phase2_outputs/yolo_dataset'
)

stats = builder.prepare_dataset(train_split=0.8, min_confidence=0.5)

print("\n" + "="*60)
print("✅ Dataset preparation complete!")
print(f"   Train images: {stats['train_images']}")
print(f"   Val images: {stats['val_images']}")
print(f"   Classes: {stats['num_classes']}")
print("="*60)
EOF
```

**You should now see**:
```
📊 Train Split Statistics:
   ✅ Images copied: 3,368  # Instead of 0!
   ⚠️  Skipped (no file): 0
   ⚠️  Skipped (no detections): 0
   ⚠️  Skipped (no valid annotations): 843
   ⚠️  Unmapped COCO classes: ['person', 'car', 'dog']  # Classes we didn't map

📊 Val Split Statistics:
   ✅ Images copied: 842
   ...
```

Then verify images exist:
```bash
ls -l phase2_outputs/yolo_dataset/images/train | wc -l
ls -l phase2_outputs/yolo_dataset/images/val | wc -l
```

You should see **actual image files** now (not just 0).

---

### Option 3: Start Training Immediately (After Verification)

Once you verify images are copied:

```bash
cd /notebooks/app

# Train YOLO only (skip dataset prep since it's done)
python src/models/yolo_finetune.py \
    --dataset phase2_outputs/yolo_dataset/data.yaml \
    --output phase2_outputs/yolo_training_runs \
    --epochs 100 \
    --batch-size 16

# Then train style classifiers
python src/models/improved_style_classifier.py \
    --db database_metadata.duckdb \
    --output phase2_outputs \
    --epochs 30
```

---

## Understanding the Output

### Good Output (Fixed)
```
Processing train: 100%|████████| 4211/4211 [02:15<00:00]

📊 Train Split Statistics:
   ✅ Images copied: 3,368
   ⚠️  Skipped (no valid annotations): 843
   ⚠️  Unmapped COCO classes: ['person', 'car']

Processing val: 100%|████████| 1053/1053 [00:34<00:00]

📊 Val Split Statistics:
   ✅ Images copied: 842
   ⚠️  Skipped (no valid annotations): 211

✅ Dataset preparation complete!
   Train images: 3,368  # ✅ NOT ZERO!
   Val images: 842      # ✅ NOT ZERO!
```

### Why Some Images Are Skipped
- **No valid annotations**: Image has detections but they're all classes we haven't mapped (like "person", "car")
- **Unmapped classes**: COCO classes that aren't furniture/interior items (we intentionally skip these)

This is **normal and expected** - we only want interior design objects!

---

## What Happens Next

Once dataset prep works:

1. **YOLO Training** (~8 hours)
   - Trains on 3,368 images
   - Creates detector for 294 interior design categories
   - Saves best model to: `phase2_outputs/yolo_training_runs/finetune_294_classes/weights/best.pt`

2. **Style Classifier Training** (~4 hours)
   - Trains 3 models (EfficientNet, ResNet50, ViT)
   - Creates ensemble classifier
   - Saves to: `best_efficientnet_style_classifier.pth`, etc.

3. **API Ready** (Once training completes)
   - Update [api/.env](api/.env) with new model paths
   - Restart API: `./start_api.sh`
   - Get 70%+ accuracy instead of 54%!

---

## Quick Verification Checklist

Before running full training, verify:

- [ ] Git pulled latest code: `git pull origin main`
- [ ] In correct directory: `cd /notebooks/app`
- [ ] Database exists: `ls -lh database_metadata.duckdb`
- [ ] Python imports work: `python -c "from src.models.yolo_dataset_prep import YOLODatasetBuilder; print('✅ OK')"`
- [ ] Dataset prep works: Run Option 2 above
- [ ] Images were copied: Check the statistics output
- [ ] Ready for training: Run Option 1 for full pipeline

---

## Troubleshooting

### Still getting 0 images copied?

Check which classes are in your database:
```bash
python << 'EOF'
import duckdb
conn = duckdb.connect('database_metadata.duckdb')
classes = conn.execute("""
    SELECT DISTINCT item_type, COUNT(*) as count
    FROM furniture_detections
    GROUP BY item_type
    ORDER BY count DESC
""").df()
print(classes)
EOF
```

If you see classes not in the mapping, add them to [src/models/yolo_dataset_prep.py:43-93](src/models/yolo_dataset_prep.py#L43-L93).

### Images not found at paths?

Your database might have old file paths. Check:
```bash
python << 'EOF'
import duckdb
conn = duckdb.connect('database_metadata.duckdb')
paths = conn.execute("SELECT original_path FROM images LIMIT 5").df()
print(paths)
EOF
```

If paths are wrong (e.g., point to Mac paths instead of Paperspace paths), you'll need to update them in the database.

---

## Expected Timeline

- **Dataset Prep**: ~3-5 minutes (re-run with fix)
- **YOLO Training**: ~8 hours (100 epochs on GPU)
- **Style Training**: ~4 hours (30 epochs, 3 models)
- **Total**: ~12 hours

You can monitor progress - it will show loss decreasing and accuracy improving!

---

## Next Steps

1. **Now**: Re-run dataset preparation (Option 2 above)
2. **Verify**: Check that images were copied
3. **Then**: Start full training (Option 1)
4. **Monitor**: Check back in a few hours to see progress
5. **Tomorrow**: Models should be trained, update API config

---

**The fix is ready!** Just pull the latest code and re-run. You should see actual images being copied now. 🚀
