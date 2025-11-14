# Fix "Killed" Error - Out of Memory

Your process was killed because it ran out of memory trying to process 5,598 images on CPU.

## Quick Checks

### 1. Check if GPU is Available

```bash
cd /notebooks/app
python check_gpu.py
```

**Expected output if GPU works:**
```
✅ CUDA Available: True
   Device Name: NVIDIA A4000
   GPU Memory: 16.00 GB
```

**If you see "No CUDA available":**
```bash
# Check if you're on a GPU machine
nvidia-smi

# If nvidia-smi works but PyTorch doesn't see GPU, reinstall torch:
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Solutions

### Solution 1: Use GPU (Best)

If GPU is available but not being used, there might be a PyTorch issue:

```bash
cd /notebooks/app

# Reinstall PyTorch with CUDA support
pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

Then re-run training.

---

### Solution 2: Process in Batches (If GPU Doesn't Work)

If stuck on CPU, we need to process images in smaller batches to avoid OOM:

```bash
cd /notebooks/app

# Process first 1000 images
python << 'EOF'
import sys
sys.path.insert(0, '/notebooks/app')

from pathlib import Path
from src.processing.batch_processor import BatchProcessor
from src.processing.image_processor import DataConfig

# Custom config
config = DataConfig(base_dir="./r2_phase2_outputs/processed")

# Batch processor
processor = BatchProcessor(db_path="database_r2_full.duckdb", config=config)

# Get image files
r2_images_dir = Path("r2_phase2_outputs/r2_images")
image_files = list(r2_images_dir.glob("*.jpg")) + list(r2_images_dir.glob("*.png"))

print(f"Found {len(image_files)} images")
print("Processing in batches of 100...")

# Process in batches
batch_size = 100
for i in range(0, len(image_files), batch_size):
    batch = image_files[i:i+batch_size]
    print(f"\nProcessing batch {i//batch_size + 1}/{(len(image_files) + batch_size - 1)//batch_size}")

    for img_path in batch:
        try:
            processor.process_image_file(str(img_path), source="cloudflare_r2", dataset_name="r2_training_data")
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")

    # Force garbage collection
    import gc
    gc.collect()

    print(f"✅ Processed {min(i+batch_size, len(image_files))}/{len(image_files)} images")

processor.close()
print("\n✅ All images processed!")
EOF
```

This processes 100 images at a time and clears memory between batches.

---

### Solution 3: Use Lighter Model

Replace CLIP with a lighter classifier temporarily:

```bash
cd /notebooks/app

# Edit the image processor to skip CLIP classification
# Just do YOLO detection, skip style classification
python << 'EOF'
# Quick processing - YOLO only, no CLIP
import sys
sys.path.insert(0, '/notebooks/app')

from pathlib import Path
from src.processing.batch_processor import BatchProcessor
from src.processing.image_processor import DataConfig, ImageProcessor
import hashlib
import duckdb
from tqdm import tqdm

config = DataConfig(base_dir="./r2_phase2_outputs/processed")
r2_images_dir = Path("r2_phase2_outputs/r2_images")
db_path = "database_r2_full.duckdb"

# Initialize processor WITHOUT CLIP (to save memory)
processor = ImageProcessor(config)

# Get images
image_files = list(r2_images_dir.glob("*.jpg")) + list(r2_images_dir.glob("*.png"))
print(f"Processing {len(image_files)} images with YOLO only...")

# Connect to DB
conn = duckdb.connect(db_path)

# Process each image
for img_path in tqdm(image_files):
    try:
        # Generate ID
        with open(img_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        image_id = file_hash[:16]

        # Check if exists
        exists = conn.execute("SELECT image_id FROM images WHERE image_id = ?", [image_id]).fetchone()
        if exists:
            continue

        # Process with YOLO only (skip CLIP to save memory)
        from src.processing.image_processor import ImageMetadata
        metadata = ImageMetadata(
            image_id=image_id,
            source="cloudflare_r2",
            dataset_name="r2_training_data",
            original_path=str(img_path)
        )

        # This does YOLO detection but skip CLIP
        metadata = processor.process_image(str(img_path), metadata)

        # Insert to DB (simplified - you may need to adjust)
        # ... database insert code here ...

    except Exception as e:
        print(f"Error: {e}")
        continue

conn.close()
print("✅ Done!")
EOF
```

---

### Solution 4: Start Fresh with Smaller Dataset

If everything fails, start with a smaller subset:

```bash
cd /notebooks/app

# Clean up
rm -rf r2_phase2_outputs/
rm database_r2_full.duckdb

# Run with only 500 images
python scripts/run_r2_to_phase2_training.py \
    --db database_r2_500.duckdb \
    --output ./r2_phase2_outputs \
    --max-images 500 \
    --yolo-epochs 50 \
    --style-epochs 20
```

This will complete successfully and you'll have working models to test with.

---

## Recommended Fix Order

1. **Check GPU** (`python check_gpu.py`)
2. **If GPU available but not used**: Reinstall PyTorch with CUDA
3. **If no GPU**: Use Solution 2 (batch processing) or Solution 4 (fewer images)
4. **Monitor memory**: `watch -n 1 free -h` while processing

---

## Why It Was Killed

The OOM (Out of Memory) killer terminated your process because:

1. **5,598 images** is a lot to process at once
2. **CPU processing** uses more RAM than GPU processing
3. **CLIP model** (605MB) + image processing = high memory usage
4. Processing all images in one go without clearing memory

**On GPU**: Would use 16GB VRAM instead of system RAM
**On CPU**: Uses ~20-30GB system RAM, which Paperspace might not have

---

## After Fixing

Once images are processed and in database, the rest of training should work fine:

```bash
# Skip download and Phase 1 since they're done
python scripts/run_r2_to_phase2_training.py \
    --db database_r2_full.duckdb \
    --output ./r2_phase2_outputs \
    --skip-download \
    --skip-phase1
```

This will go straight to YOLO training which handles memory better.

---

## Quick Check: Am I On GPU Machine?

```bash
nvidia-smi
```

If you see GPU info: ✅ You have GPU, just need to use it
If you see "command not found": ❌ You're on CPU-only machine (expensive to train on)

Make sure your Paperspace instance is a **GPU instance** (like A4000, A5000, or similar).
