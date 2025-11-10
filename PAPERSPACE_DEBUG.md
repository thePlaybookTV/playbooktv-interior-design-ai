# Debug Image Paths Issue

## Run This in Paperspace

```bash
cd /notebooks/app

# Pull latest debug script
git pull origin main

# Make it executable
chmod +x debug_database.py

# Run the debug script
python debug_database.py
```

This will show:
1. What detection classes are in the database
2. Sample image paths from the database
3. Whether those paths actually exist on the Paperspace filesystem
4. Where image files might actually be located

---

## What We're Looking For

The database likely has paths like:
- `/Users/leslieisah/...` (Mac paths - won't work on Paperspace)
- `./images/...` (relative paths - might work if images are uploaded)
- `/notebooks/...` (Paperspace paths - would work)

**The problem**: The database was created on your Mac, so it has Mac file paths. Those files don't exist on Paperspace.

---

## Quick Alternative: Check Database Now

```bash
cd /notebooks/app

# Quick check - see what paths are in database
python << 'EOF'
import duckdb
conn = duckdb.connect('database_metadata.duckdb')

# Show sample paths
print("Sample paths from database:")
paths = conn.execute("SELECT original_path FROM images LIMIT 5").fetchall()
for (path,) in paths:
    print(f"  {path}")

# Show detection classes
print("\nDetection classes:")
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

---

## Likely Solutions

### Solution 1: Images Were Never Uploaded to Paperspace
If the database has image metadata but the actual image files weren't uploaded:

**You need to upload the images to Paperspace.**

Where are your images? Check on your Mac:
```bash
# On your Mac (locally)
find ~/MDMv3/playbooktv-interior-design-ai -name "*.jpg" -o -name "*.png" | head -20
```

Then upload them to Paperspace via the file browser.

---

### Solution 2: Update Database Paths
If images are in Paperspace but paths in database are wrong:

```bash
cd /notebooks/app

# Update paths in database
python << 'EOF'
import duckdb
import os
from pathlib import Path

conn = duckdb.connect('database_metadata.duckdb')

# Find where images actually are
image_dir = None
for possible_dir in ['/notebooks/images', '/notebooks/data', '/notebooks/app/images']:
    if os.path.exists(possible_dir):
        files = list(Path(possible_dir).glob('*.jpg'))
        if files:
            image_dir = possible_dir
            print(f"✅ Found images in: {image_dir}")
            break

if image_dir:
    # Update paths (example - adjust based on your structure)
    conn.execute(f"""
        UPDATE images
        SET original_path = '{image_dir}/' || filename
        WHERE filename IS NOT NULL
    """)
    print("✅ Paths updated")
else:
    print("❌ No images found on Paperspace")

conn.close()
EOF
```

---

### Solution 3: Re-download Images
If you scraped images from Unsplash/Pexels, you can re-download them:

```bash
cd /notebooks/app

# Re-run the scraping (if you have the script)
python src/data_collection/scrape_images.py

# This will populate images/ directory and update database paths
```

---

## After Fixing Paths

Once images are accessible, re-run dataset prep:

```bash
cd /notebooks/app

# Clean old attempt
rm -rf phase2_outputs/yolo_dataset

# Re-prepare dataset
python << 'EOF'
import sys
sys.path.insert(0, '/notebooks/app')

from src.models.yolo_dataset_prep import YOLODatasetBuilder

builder = YOLODatasetBuilder(
    db_path='database_metadata.duckdb',
    output_dir='phase2_outputs/yolo_dataset'
)

stats = builder.prepare_dataset(train_split=0.8, min_confidence=0.5)
print(f"\n✅ Train images: {stats['train_images']}")
print(f"✅ Val images: {stats['val_images']}")
EOF

# Verify images were copied
ls -lh phase2_outputs/yolo_dataset/images/train | head -10
```

---

## First Step: Run the Debug

**Copy this into Paperspace:**

```bash
cd /notebooks/app
python << 'EOF'
import duckdb
import os

conn = duckdb.connect('database_metadata.duckdb')

# Check sample paths
print("="*60)
print("SAMPLE PATHS FROM DATABASE:")
print("="*60)
paths = conn.execute("SELECT original_path FROM images LIMIT 10").fetchall()
for (path,) in paths:
    exists = "✅" if path and os.path.exists(path) else "❌"
    print(f"{exists} {path}")

# Check if images exist at all
print("\n" + "="*60)
print("CHECKING FOR IMAGE FILES ON PAPERSPACE:")
print("="*60)

import glob
for search_dir in ['/notebooks', '/notebooks/app', '.']:
    if os.path.exists(search_dir):
        jpg_files = glob.glob(f"{search_dir}/**/*.jpg", recursive=True)
        png_files = glob.glob(f"{search_dir}/**/*.png", recursive=True)
        total = len(jpg_files) + len(png_files)
        if total > 0:
            print(f"📂 {search_dir}: {total} image files")
            # Show first few
            for f in (jpg_files + png_files)[:3]:
                print(f"   {f}")

print("\n" + "="*60)
print("DETECTION CLASSES:")
print("="*60)
classes = conn.execute("""
    SELECT item_type, COUNT(*) as count
    FROM furniture_detections
    GROUP BY item_type
    ORDER BY count DESC
    LIMIT 15
""").df()
print(classes)

conn.close()
EOF
```

**Send me the output** and I'll tell you exactly what to do next.
