#!/usr/bin/env python3
"""
Debug script to investigate database contents and image paths
"""

import duckdb
import os
from pathlib import Path

print("="*60)
print("🔍 DEBUGGING DATABASE AND IMAGE PATHS")
print("="*60)

# Connect to database
db_path = 'database_metadata.duckdb'
if not os.path.exists(db_path):
    print(f"❌ Database not found at: {db_path}")
    exit(1)

conn = duckdb.connect(db_path)

# 1. Check what detection classes exist
print("\n1️⃣  DETECTION CLASSES IN DATABASE")
print("-" * 60)
classes = conn.execute("""
    SELECT item_type, COUNT(*) as count
    FROM furniture_detections
    GROUP BY item_type
    ORDER BY count DESC
    LIMIT 20
""").df()
print(classes.to_string())

# 2. Check sample image paths
print("\n\n2️⃣  SAMPLE IMAGE PATHS IN DATABASE")
print("-" * 60)
sample_paths = conn.execute("""
    SELECT original_path
    FROM images
    LIMIT 10
""").df()
print(sample_paths.to_string())

# 3. Check if any paths exist on filesystem
print("\n\n3️⃣  CHECKING IF PATHS EXIST ON FILESYSTEM")
print("-" * 60)
paths_to_check = conn.execute("""
    SELECT original_path
    FROM images
    LIMIT 100
""").fetchall()

exists_count = 0
not_exists_count = 0
sample_existing = []
sample_missing = []

for (path,) in paths_to_check:
    if path and os.path.exists(path):
        exists_count += 1
        if len(sample_existing) < 3:
            sample_existing.append(path)
    else:
        not_exists_count += 1
        if len(sample_missing) < 3:
            sample_missing.append(path)

print(f"✅ Paths that exist: {exists_count}")
print(f"❌ Paths that don't exist: {not_exists_count}")

if sample_existing:
    print("\n📂 Sample existing paths:")
    for p in sample_existing:
        print(f"   {p}")

if sample_missing:
    print("\n❌ Sample missing paths:")
    for p in sample_missing:
        print(f"   {p}")

# 4. Check images with detections
print("\n\n4️⃣  IMAGES WITH DETECTIONS")
print("-" * 60)
images_with_detections = conn.execute("""
    SELECT
        COUNT(DISTINCT i.image_id) as total_images,
        COUNT(DISTINCT fd.detection_id) as total_detections
    FROM images i
    INNER JOIN furniture_detections fd ON i.image_id = fd.image_id
""").df()
print(images_with_detections.to_string())

# 5. Check specific detections with confidence
print("\n\n5️⃣  DETECTION CONFIDENCE LEVELS")
print("-" * 60)
confidence_stats = conn.execute("""
    SELECT
        item_type,
        COUNT(*) as count,
        AVG(confidence) as avg_confidence,
        MIN(confidence) as min_confidence,
        MAX(confidence) as max_confidence
    FROM furniture_detections
    GROUP BY item_type
    ORDER BY count DESC
    LIMIT 10
""").df()
print(confidence_stats.to_string())

# 6. Check image + detection join with paths
print("\n\n6️⃣  SAMPLE IMAGES WITH DETECTIONS AND PATHS")
print("-" * 60)
sample_data = conn.execute("""
    SELECT
        i.image_id,
        i.original_path,
        fd.item_type,
        fd.confidence,
        fd.bbox_x1, fd.bbox_y1, fd.bbox_x2, fd.bbox_y2
    FROM images i
    INNER JOIN furniture_detections fd ON i.image_id = fd.image_id
    WHERE fd.confidence >= 0.5
    LIMIT 5
""").df()
print(sample_data.to_string())

# 7. Check if paths are relative or absolute
print("\n\n7️⃣  PATH ANALYSIS")
print("-" * 60)
all_paths = conn.execute("SELECT original_path FROM images LIMIT 100").fetchall()
relative_paths = sum(1 for (p,) in all_paths if p and not os.path.isabs(p))
absolute_paths = sum(1 for (p,) in all_paths if p and os.path.isabs(p))
null_paths = sum(1 for (p,) in all_paths if not p)

print(f"Relative paths: {relative_paths}")
print(f"Absolute paths: {absolute_paths}")
print(f"Null/empty paths: {null_paths}")

# 8. Try to find where images might actually be
print("\n\n8️⃣  SEARCHING FOR IMAGE FILES")
print("-" * 60)
common_dirs = [
    '/notebooks',
    '/notebooks/app',
    '/notebooks/images',
    '/notebooks/data',
    'images',
    'data',
    'downloaded_images',
    '.'
]

for dir_path in common_dirs:
    if os.path.exists(dir_path):
        # Count image files
        try:
            image_files = list(Path(dir_path).rglob('*.jpg')) + \
                         list(Path(dir_path).rglob('*.png')) + \
                         list(Path(dir_path).rglob('*.jpeg'))
            if image_files:
                print(f"📂 {dir_path}: Found {len(image_files)} image files")
                if len(image_files) <= 5:
                    for img in image_files[:5]:
                        print(f"   {img}")
        except Exception as e:
            print(f"⚠️  {dir_path}: Error scanning - {e}")

print("\n" + "="*60)
print("✅ DEBUG COMPLETE")
print("="*60)

conn.close()
