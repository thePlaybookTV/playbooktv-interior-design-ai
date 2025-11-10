#!/usr/bin/env python3
"""
Complete R2-to-Phase2 Training Pipeline

This script:
1. Downloads images from Cloudflare R2
2. Processes them with Phase 1 (YOLO detection, style classification)
3. Stores metadata in DuckDB
4. Runs Phase 2 training (fine-tune YOLO, train ensemble classifiers)
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_collection.cloudflare_r2_downloader import CloudflareR2Downloader
from src.processing.batch_processor import BatchProcessor
from src.models.yolo_dataset_prep import YOLODatasetBuilder
from src.models.yolo_finetune import YOLOFineTuner
from src.models.improved_style_classifier import ImprovedStyleClassifier


class R2ToPhase2Pipeline:
    """Complete pipeline from R2 images to trained Phase 2 models"""

    def __init__(self, db_path: str, output_dir: str):
        self.db_path = db_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.r2_images_dir = self.output_dir / "r2_images"
        self.yolo_dataset_dir = self.output_dir / "yolo_dataset"
        self.yolo_training_dir = self.output_dir / "yolo_training_runs"

        print("="*80)
        print("🚀 R2-TO-PHASE2 TRAINING PIPELINE")
        print("="*80)
        print(f"Database: {self.db_path}")
        print(f"Output Directory: {self.output_dir}")
        print(f"Timestamp: {datetime.now().strftime('%Y%m%d_%H%M%S')}")
        print("="*80)

    def step1_download_from_r2(
        self,
        prefix: str = "",
        max_images: int = None,
        max_workers: int = 20
    ):
        """Step 1: Download images from Cloudflare R2"""
        print("\n" + "="*80)
        print("📥 STEP 1: DOWNLOADING IMAGES FROM R2")
        print("="*80)

        try:
            # Initialize R2 downloader
            downloader = CloudflareR2Downloader()

            # Test connection
            if not downloader.test_connection():
                raise Exception("Failed to connect to R2")

            # Download all images
            downloaded = downloader.download_all_images(
                output_dir=self.r2_images_dir,
                prefix=prefix,
                keep_structure=False,  # Flatten to one directory
                max_workers=max_workers,
                skip_existing=True,
                max_images=max_images
            )

            if not downloaded:
                raise Exception("No images were downloaded from R2")

            print(f"\n✅ Downloaded {len(downloaded)} images to {self.r2_images_dir}")
            return len(downloaded)

        except Exception as e:
            print(f"\n❌ Error downloading from R2: {e}")
            print("\nMake sure you have these environment variables set:")
            print("  - CLOUDFLARE_ACCOUNT_ID")
            print("  - R2_ACCESS_KEY_ID")
            print("  - R2_SECRET_ACCESS_KEY")
            print("  - R2_BUCKET_NAME")
            raise

    def step2_process_images_phase1(self):
        """Step 2: Process images with Phase 1 (YOLO + Classification)"""
        print("\n" + "="*80)
        print("🔧 STEP 2: PROCESSING IMAGES (PHASE 1)")
        print("="*80)
        print("This will:")
        print("  - Run YOLO detection on all images")
        print("  - Classify room types and styles")
        print("  - Store detections in DuckDB")
        print("="*80)

        # Initialize batch processor
        processor = BatchProcessor(db_path=self.db_path)

        # Process all images in R2 directory
        count = processor.process_directory(
            directory=self.r2_images_dir,
            source="cloudflare_r2",
            dataset_name="r2_training_data"
        )

        processor.close()

        print(f"\n✅ Phase 1 processing complete! Processed {count} images")
        return count

    def step3_prepare_yolo_dataset(self, train_split: float = 0.8):
        """Step 3: Prepare YOLO dataset from DuckDB"""
        print("\n" + "="*80)
        print("📦 STEP 3: PREPARING YOLO DATASET")
        print("="*80)

        builder = YOLODatasetBuilder(
            db_path=self.db_path,
            output_dir=str(self.yolo_dataset_dir)
        )

        stats = builder.prepare_dataset(train_split=train_split)

        print(f"\n✅ YOLO dataset preparation complete!")
        print(f"   Train images: {stats['train_images']}")
        print(f"   Val images: {stats['val_images']}")
        print(f"   Total classes: {stats['num_classes']}")

        return stats

    def step4_finetune_yolo(
        self,
        epochs: int = 100,
        batch_size: int = 16,
        imgsz: int = 640
    ):
        """Step 4: Fine-tune YOLO on 294 categories"""
        print("\n" + "="*80)
        print("🎯 STEP 4: FINE-TUNING YOLO ON 294 CATEGORIES")
        print("="*80)

        data_yaml = self.yolo_dataset_dir / "data.yaml"

        if not data_yaml.exists():
            raise FileNotFoundError(f"YOLO data.yaml not found at {data_yaml}")

        finetuner = YOLOFineTuner(
            data_yaml=str(data_yaml),
            output_dir=str(self.yolo_training_dir)
        )

        train_results, val_results = finetuner.train(
            epochs=epochs,
            batch_size=batch_size,
            imgsz=imgsz,
            lr0=0.01,
            device='auto'  # Auto-detect GPU
        )

        print(f"\n✅ YOLO fine-tuning complete!")
        print(f"   Best model: {self.yolo_training_dir}/finetune_294_classes/weights/best.pt")

        return train_results, val_results

    def step5_train_style_classifiers(self, epochs: int = 30):
        """Step 5: Train improved style classifiers"""
        print("\n" + "="*80)
        print("🎨 STEP 5: TRAINING IMPROVED STYLE CLASSIFIERS")
        print("="*80)

        classifier = ImprovedStyleClassifier(
            db_path=self.db_path,
            output_dir=str(self.output_dir)
        )

        # This will train EfficientNet, ResNet50, and ViT
        results = classifier.train_all_models(epochs=epochs)

        print(f"\n✅ Style classifier training complete!")

        return results

    def run_full_pipeline(
        self,
        r2_prefix: str = "",
        max_images: int = None,
        train_split: float = 0.8,
        yolo_epochs: int = 100,
        yolo_batch_size: int = 16,
        style_epochs: int = 30,
        skip_download: bool = False,
        skip_phase1: bool = False
    ):
        """Run the complete pipeline"""

        try:
            # Step 1: Download from R2
            if not skip_download:
                image_count = self.step1_download_from_r2(
                    prefix=r2_prefix,
                    max_images=max_images
                )
            else:
                print("\n⏭️  Skipping R2 download (--skip-download)")
                # Count existing images
                image_files = list(self.r2_images_dir.glob("*.jpg")) + \
                             list(self.r2_images_dir.glob("*.png"))
                image_count = len(image_files)
                print(f"   Found {image_count} existing images")

            if image_count == 0:
                raise Exception("No images available for processing")

            # Step 2: Process with Phase 1
            if not skip_phase1:
                processed_count = self.step2_process_images_phase1()
                if processed_count == 0:
                    raise Exception("No images were processed in Phase 1")
            else:
                print("\n⏭️  Skipping Phase 1 processing (--skip-phase1)")

            # Step 3: Prepare YOLO dataset
            dataset_stats = self.step3_prepare_yolo_dataset(train_split=train_split)

            if dataset_stats['train_images'] == 0:
                raise Exception("No training images in YOLO dataset!")

            # Step 4: Fine-tune YOLO
            train_results, val_results = self.step4_finetune_yolo(
                epochs=yolo_epochs,
                batch_size=yolo_batch_size
            )

            # Step 5: Train style classifiers
            style_results = self.step5_train_style_classifiers(epochs=style_epochs)

            # Summary
            print("\n" + "="*80)
            print("✅ PIPELINE COMPLETE!")
            print("="*80)
            print(f"\n📊 Summary:")
            print(f"   Images downloaded: {image_count}")
            print(f"   Training images: {dataset_stats['train_images']}")
            print(f"   Validation images: {dataset_stats['val_images']}")
            print(f"   YOLO classes: {dataset_stats['num_classes']}")
            print(f"\n📁 Output files:")
            print(f"   YOLO model: {self.yolo_training_dir}/finetune_294_classes/weights/best.pt")
            print(f"   Style classifiers: {self.output_dir}/best_*_style_classifier.pth")
            print(f"   Database: {self.db_path}")
            print("\n🚀 Ready to deploy to API!")
            print("="*80)

            return {
                'image_count': image_count,
                'dataset_stats': dataset_stats,
                'yolo_results': (train_results, val_results),
                'style_results': style_results
            }

        except Exception as e:
            print(f"\n❌ Pipeline failed: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description='Complete R2-to-Phase2 training pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python scripts/run_r2_to_phase2_training.py --db database.duckdb --output ./outputs

  # Download only 1000 images for testing
  python scripts/run_r2_to_phase2_training.py --db database.duckdb --max-images 1000

  # Skip download if images already downloaded
  python scripts/run_r2_to_phase2_training.py --db database.duckdb --skip-download

  # Quick training run (fewer epochs)
  python scripts/run_r2_to_phase2_training.py --db database.duckdb --yolo-epochs 10 --style-epochs 5
        """
    )

    parser.add_argument('--db', type=str, required=True,
                        help='Path to DuckDB database')
    parser.add_argument('--output', '-o', type=str, default='./phase2_outputs',
                        help='Output directory for all files')

    # R2 download options
    parser.add_argument('--r2-prefix', type=str, default='',
                        help='R2 bucket prefix/folder to download from')
    parser.add_argument('--max-images', type=int, default=None,
                        help='Maximum number of images to download from R2')

    # Training options
    parser.add_argument('--train-split', type=float, default=0.8,
                        help='Train/val split ratio (default: 0.8)')
    parser.add_argument('--yolo-epochs', type=int, default=100,
                        help='Number of YOLO training epochs')
    parser.add_argument('--yolo-batch-size', type=int, default=16,
                        help='YOLO training batch size')
    parser.add_argument('--style-epochs', type=int, default=30,
                        help='Number of style classifier training epochs')

    # Skip options
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip R2 download (use existing images)')
    parser.add_argument('--skip-phase1', action='store_true',
                        help='Skip Phase 1 processing (use existing DB)')

    args = parser.parse_args()

    # Create pipeline
    pipeline = R2ToPhase2Pipeline(
        db_path=args.db,
        output_dir=args.output
    )

    # Run pipeline
    results = pipeline.run_full_pipeline(
        r2_prefix=args.r2_prefix,
        max_images=args.max_images,
        train_split=args.train_split,
        yolo_epochs=args.yolo_epochs,
        yolo_batch_size=args.yolo_batch_size,
        style_epochs=args.style_epochs,
        skip_download=args.skip_download,
        skip_phase1=args.skip_phase1
    )

    print("\n✅ All done!")


if __name__ == "__main__":
    main()
