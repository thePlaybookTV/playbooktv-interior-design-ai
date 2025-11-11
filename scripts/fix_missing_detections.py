#!/usr/bin/env python3
"""
Emergency Fix: Add detections to already-processed images
This will run YOLO+SAM2 detection on images that are already in the database
but don't have any furniture detections yet.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.pristine_detector import CheckpointProcessor
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='Add furniture detections to already-processed images'
    )
    parser.add_argument('--db', type=str, required=True,
                        help='Path to DuckDB database')

    args = parser.parse_args()

    print("=" * 80)
    print("🔧 EMERGENCY FIX: Adding Missing Detections")
    print("=" * 80)
    print(f"Database: {args.db}")
    print("\nThis will:")
    print("  - Find all images without detections")
    print("  - Run YOLO+SAM2 on each image")
    print("  - Save detections to furniture_detections table")
    print("=" * 80)

    # Initialize processor
    processor = CheckpointProcessor(args.db)

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
    processor.show_stats()

    processor.close()

    print("\n" + "=" * 80)
    print("✅ DETECTION FIX COMPLETE!")
    print("=" * 80)
    print("\n🎉 Your database now has furniture detections!")
    print("   You can now run Phase 2 training:")
    print(f"   python scripts/run_phase2_training.py --db {args.db}")
    print("=" * 80)


if __name__ == "__main__":
    main()
