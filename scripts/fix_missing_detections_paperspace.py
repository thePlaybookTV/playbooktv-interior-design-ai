#!/usr/bin/env python3
"""
Emergency Fix for Paperspace: Add detections to already-processed images
This will run YOLO+SAM2 detection on images that are already in the database
but don't have any furniture detections yet.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set working directory to project root
os.chdir(project_root)
print(f"📁 Working directory: {os.getcwd()}")

# Set model paths explicitly for Gradient Paperspace
# Models are mounted in /datasets/
os.environ['YOLO_MODEL_PATH'] = '/datasets/yolo/yolov8m.pt'
os.environ['SAM2_CHECKPOINT'] = '/datasets/sam2/sam2_hiera_large.pt'
os.environ['SAM2_CONFIG'] = 'sam2_hiera_l.yaml'

print(f"🔧 YOLO model: {os.environ['YOLO_MODEL_PATH']}")
print(f"🔧 SAM2 checkpoint: {os.environ['SAM2_CHECKPOINT']}")

from src.models.pristine_detector import CheckpointProcessor
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='Add furniture detections to already-processed images (Paperspace version)'
    )
    parser.add_argument('--db', type=str, required=True,
                        help='Path to DuckDB database')

    args = parser.parse_args()

    # Convert to absolute path
    db_path = Path(args.db)
    if not db_path.is_absolute():
        db_path = project_root / db_path

    print("=" * 80)
    print("🔧 EMERGENCY FIX: Adding Missing Detections (Paperspace)")
    print("=" * 80)
    print(f"Database: {db_path}")
    print("\nThis will:")
    print("  - Find all images without detections")
    print("  - Run YOLO+SAM2 on each image")
    print("  - Save detections to furniture_detections table")
    print("=" * 80)

    # Verify models exist
    print("\n🔍 Verifying models...")
    yolo_path = Path(os.environ['YOLO_MODEL_PATH'])
    sam2_path = Path(os.environ['SAM2_CHECKPOINT'])

    if not yolo_path.exists():
        print(f"❌ YOLO model not found: {yolo_path}")
        print(f"   Please ensure Gradient dataset is mounted correctly")
        return
    else:
        print(f"✅ YOLO model found: {yolo_path}")

    if not sam2_path.exists():
        print(f"❌ SAM2 checkpoint not found: {sam2_path}")
        print(f"   Please ensure Gradient dataset is mounted correctly")
        return
    else:
        print(f"✅ SAM2 checkpoint found: {sam2_path}")

    # Clear old checkpoint (it's from the broken run without detections)
    checkpoint_file = project_root / "processing_checkpoint.json"
    if checkpoint_file.exists():
        print("\n🗑️  Removing old checkpoint file (from broken run)...")
        checkpoint_file.unlink()
        print("   ✅ Checkpoint cleared - will reprocess all images")

    # Initialize processor
    processor = CheckpointProcessor(str(db_path))

    # Check if detector initialized properly
    if not processor.detector.yolo or not processor.detector.sam2:
        print("\n❌ Detector failed to initialize!")
        print("   YOLO loaded:", processor.detector.yolo is not None)
        print("   SAM2 loaded:", processor.detector.sam2 is not None)
        processor.close()
        return

    # Process all images
    print("\n🚀 Starting detection processing...")
    processor.process_all(
        batch_size=50,  # Smaller batches for safety
        checkpoint_interval=100  # Save progress every 100 images
    )

    # Show statistics
    print("\n" + "=" * 80)
    print("📊 FINAL STATISTICS")
    print("=" * 80)

    try:
        processor.show_stats()
    except ZeroDivisionError:
        print("⚠️  No detections were created - models may not have loaded correctly")
        print("\nDebugging info:")
        print(f"  YOLO model: {os.environ.get('YOLO_MODEL_PATH')}")
        print(f"  SAM2 checkpoint: {os.environ.get('SAM2_CHECKPOINT')}")

    processor.close()

    print("\n" + "=" * 80)
    print("✅ DETECTION FIX COMPLETE!")
    print("=" * 80)
    print("\n🎉 Check the statistics above to verify detections were created.")
    print("   If detections > 0, you can now run Phase 2 training:")
    print(f"   python scripts/run_phase2_training.py --db {db_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
