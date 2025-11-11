#!/usr/bin/env python3
"""
Process R2 images in memory-safe batches
Avoids OOM by processing smaller chunks at a time
"""

import os
import sys
import gc
import argparse
from pathlib import Path
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.processing.batch_processor_with_sam2 import BatchProcessorWithSAM2
from src.processing.image_processor import DataConfig


def process_in_batches(
    images_dir: str,
    db_path: str,
    batch_size: int = 50,
    output_dir: str = "./processed"
):
    """
    Process images in small batches to avoid OOM

    Args:
        images_dir: Directory with images to process
        db_path: Path to DuckDB database
        batch_size: Number of images per batch (lower = less memory)
        output_dir: Output directory for processed data
    """

    print("="*70)
    print("🔄 BATCH IMAGE PROCESSOR")
    print("="*70)

    images_dir = Path(images_dir)

    # Get all images
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.webp', '.JPG', '.JPEG', '.PNG']:
        image_files.extend(images_dir.rglob(f"*{ext}"))

    print(f"\n📂 Found {len(image_files)} images in {images_dir}")
    print(f"📦 Processing in batches of {batch_size}")
    print(f"💾 Database: {db_path}")
    print(f"📁 Output: {output_dir}")

    if len(image_files) == 0:
        print("❌ No images found!")
        return 0

    # Create config
    config = DataConfig(base_dir=output_dir)

    # Initialize processor with YOLO+SAM2 detection
    processor = BatchProcessorWithSAM2(db_path=db_path, config=config, use_detector=True)

    # Process in batches
    total_processed = 0
    total_skipped = 0
    total_failed = 0

    num_batches = (len(image_files) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(image_files))
        batch = image_files[start_idx:end_idx]

        print(f"\n{'='*70}")
        print(f"📦 BATCH {batch_idx + 1}/{num_batches}")
        print(f"   Processing images {start_idx + 1}-{end_idx} of {len(image_files)}")
        print(f"{'='*70}")

        batch_processed = 0
        batch_failed = 0

        for img_path in tqdm(batch, desc=f"Batch {batch_idx + 1}"):
            try:
                result = processor.process_image_file(
                    str(img_path),
                    source="cloudflare_r2",
                    dataset_name="r2_training_data"
                )

                if result:
                    batch_processed += 1
                    total_processed += 1
                else:
                    total_skipped += 1

            except KeyboardInterrupt:
                print("\n⚠️  Interrupted by user")
                processor.close()
                print(f"\n📊 Processed {total_processed} images before interruption")
                return total_processed

            except Exception as e:
                batch_failed += 1
                total_failed += 1
                if batch_failed <= 3:  # Only show first 3 errors per batch
                    print(f"\n⚠️  Error processing {img_path.name}: {e}")

        print(f"\n✅ Batch {batch_idx + 1} complete:")
        print(f"   Processed: {batch_processed}")
        print(f"   Failed: {batch_failed}")

        # Force garbage collection after each batch
        gc.collect()

        # Also clear GPU cache if using CUDA
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"   🧹 Cleared GPU cache")
        except:
            pass

    # Close processor
    processor.close()

    # Final summary
    print("\n" + "="*70)
    print("✅ PROCESSING COMPLETE")
    print("="*70)
    print(f"\n📊 Summary:")
    print(f"   Total images: {len(image_files)}")
    print(f"   Successfully processed: {total_processed}")
    print(f"   Skipped (already in DB): {total_skipped}")
    print(f"   Failed: {total_failed}")
    print(f"\n💾 Database: {db_path}")

    return total_processed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Process images in memory-safe batches',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process with default batch size (50)
  python scripts/process_images_in_batches.py \\
      --images r2_phase2_outputs/r2_images \\
      --db database_r2_full.duckdb

  # Smaller batches for low memory
  python scripts/process_images_in_batches.py \\
      --images r2_phase2_outputs/r2_images \\
      --db database_r2_full.duckdb \\
      --batch-size 25

  # Larger batches if you have more memory
  python scripts/process_images_in_batches.py \\
      --images r2_phase2_outputs/r2_images \\
      --db database_r2_full.duckdb \\
      --batch-size 100
        """
    )

    parser.add_argument('--images', '-i', type=str, required=True,
                        help='Directory containing images to process')
    parser.add_argument('--db', type=str, required=True,
                        help='Path to DuckDB database')
    parser.add_argument('--batch-size', '-b', type=int, default=50,
                        help='Number of images per batch (default: 50)')
    parser.add_argument('--output', '-o', type=str, default='./processed',
                        help='Output directory for processed data')

    args = parser.parse_args()

    try:
        count = process_in_batches(
            images_dir=args.images,
            db_path=args.db,
            batch_size=args.batch_size,
            output_dir=args.output
        )

        if count > 0:
            print("\n🚀 Ready for Phase 2 training!")
            print(f"\nNext step:")
            print(f"  python scripts/run_r2_to_phase2_training.py \\")
            print(f"      --db {args.db} \\")
            print(f"      --output ./phase2_outputs \\")
            print(f"      --skip-download \\")
            print(f"      --skip-phase1")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
