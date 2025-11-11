#!/usr/bin/env python3
"""
Phase 2 Training Script - YOLO Fine-tuning & Improved Style Classification
Run this script to execute both major Phase 2 improvements
"""

import os
import sys
from pathlib import Path
import argparse
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models.yolo_dataset_prep import YOLODatasetBuilder
from models.yolo_finetune import YOLOFineTuner
from models.improved_style_classifier import train_improved_style_classifier


class Phase2Pipeline:
    """Complete Phase 2 training pipeline"""

    def __init__(self, db_path: str, output_dir: str = "./phase2_outputs"):
        self.db_path = Path(db_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        print("=" * 80)
        print(f"🚀 PHASE 2 TRAINING PIPELINE")
        print("=" * 80)
        print(f"\nDatabase: {self.db_path}")
        print(f"Output Directory: {self.output_dir}")
        print(f"Timestamp: {self.timestamp}")
        print("=" * 80)

    def step1_prepare_yolo_dataset(self):
        """Step 1: Prepare YOLO dataset from DuckDB"""

        print("\n" + "=" * 80)
        print("📦 STEP 1: PREPARING YOLO DATASET")
        print("=" * 80)

        yolo_dataset_dir = self.output_dir / "yolo_dataset"

        builder = YOLODatasetBuilder(
            db_path=str(self.db_path),
            output_dir=str(yolo_dataset_dir)
        )

        # Prepare dataset
        result = builder.prepare_dataset(
            train_split=0.8,
            min_confidence=0.5
        )

        # Get statistics
        stats = builder.get_statistics()

        builder.close()

        print("\n✅ YOLO dataset preparation complete!")
        print(f"   Train images: {stats['train']['images']:,}")
        print(f"   Val images: {stats['val']['images']:,}")
        print(f"   Total classes: {result['num_classes']}")

        return yolo_dataset_dir / 'data.yaml', result

    def step2_finetune_yolo(self, data_yaml: Path, epochs: int = 100, batch_size: int = 16):
        """Step 2: Fine-tune YOLO on 294 categories"""

        print("\n" + "=" * 80)
        print("🎯 STEP 2: FINE-TUNING YOLO ON 294 CATEGORIES")
        print("=" * 80)

        # Use YOLO model from Gradient datasets or environment variable
        yolo_model_path = os.getenv('YOLO_MODEL_PATH', '/datasets/yolo/yolov8m.pt')

        # Check if model exists, fallback to local if not
        if not Path(yolo_model_path).exists():
            print(f"⚠️  YOLO model not found at {yolo_model_path}")
            print(f"   Will download yolov8m.pt (this may take a while)")
            yolo_model_path = 'yolov8m.pt'
        else:
            print(f"✅ Using YOLO model from: {yolo_model_path}")

        finetuner = YOLOFineTuner(
            data_yaml=str(data_yaml),
            model_size=yolo_model_path
        )

        # Train
        results = finetuner.train(
            epochs=epochs,
            batch_size=batch_size,
            image_size=640,
            learning_rate=0.01,
            patience=50,
            save_period=10,
            augment=True,
            freeze_layers=0
        )

        # Validate
        val_results = finetuner.validate()

        print("\n✅ YOLO fine-tuning complete!")
        print(f"   mAP50: {val_results.box.map50:.4f}")
        print(f"   mAP50-95: {val_results.box.map:.4f}")

        return results, val_results

    def step3_train_improved_style_classifier(self, epochs: int = 30, batch_size: int = 32):
        """Step 3: Train improved style classifier with ensemble"""

        print("\n" + "=" * 80)
        print("🎨 STEP 3: TRAINING IMPROVED STYLE CLASSIFIER")
        print("=" * 80)

        ensemble, individual_results, ensemble_acc = train_improved_style_classifier(
            db_path=str(self.db_path),
            epochs=epochs,
            batch_size=batch_size
        )

        print("\n✅ Style classifier training complete!")
        print(f"   Ensemble Accuracy: {ensemble_acc:.4f}")
        print(f"   Individual Models:")
        for model_name, acc in individual_results.items():
            print(f"      {model_name}: {acc:.4f}")

        return ensemble, individual_results, ensemble_acc

    def generate_report(self, yolo_results, style_results):
        """Generate final training report"""

        print("\n" + "=" * 80)
        print("📊 GENERATING FINAL REPORT")
        print("=" * 80)

        report = {
            'timestamp': self.timestamp,
            'phase': 2,
            'yolo_training': {
                'model': 'YOLOv8m',
                'dataset': 'Custom 294-category taxonomy',
                'mAP50': float(yolo_results['val_results'].box.map50) if 'val_results' in yolo_results else 0.0,
                'mAP50-95': float(yolo_results['val_results'].box.map) if 'val_results' in yolo_results else 0.0,
            },
            'style_classification': {
                'method': 'Ensemble (EfficientNet + ResNet50 + ViT)',
                'ensemble_accuracy': style_results['ensemble_acc'],
                'individual_models': style_results['individual_results'],
                'improvement_over_phase1': style_results['ensemble_acc'] - 0.538  # Phase 1 was 53.8%
            }
        }

        report_path = self.output_dir / f'phase2_report_{self.timestamp}.json'

        import json
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n✅ Report saved: {report_path}")

        # Print summary
        print("\n" + "=" * 80)
        print("📈 PHASE 2 RESULTS SUMMARY")
        print("=" * 80)
        print(f"\n🎯 YOLO Object Detection:")
        print(f"   Model: YOLOv8m fine-tuned on 294 categories")
        print(f"   mAP50: {report['yolo_training']['mAP50']:.4f}")
        print(f"   mAP50-95: {report['yolo_training']['mAP50-95']:.4f}")

        print(f"\n🎨 Style Classification:")
        print(f"   Method: Ensemble of 3 models")
        print(f"   Ensemble Accuracy: {report['style_classification']['ensemble_accuracy']:.4f}")
        print(f"   Improvement over Phase 1: +{report['style_classification']['improvement_over_phase1']:.4f}")
        print(f"      (Phase 1: 0.538 → Phase 2: {report['style_classification']['ensemble_accuracy']:.4f})")

        print("\n" + "=" * 80)

        return report


def main():
    parser = argparse.ArgumentParser(description='Phase 2 Training Pipeline')
    parser.add_argument('--db', type=str,
                        default='./interior_design_data_hybrid/processed/metadata.duckdb',
                        help='Path to DuckDB database')
    parser.add_argument('--output', type=str,
                        default='./phase2_outputs',
                        help='Output directory')
    parser.add_argument('--skip-yolo', action='store_true',
                        help='Skip YOLO training')
    parser.add_argument('--skip-style', action='store_true',
                        help='Skip style classification training')
    parser.add_argument('--yolo-epochs', type=int, default=100,
                        help='YOLO training epochs')
    parser.add_argument('--style-epochs', type=int, default=30,
                        help='Style classifier epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for training')

    args = parser.parse_args()

    # Create pipeline
    pipeline = Phase2Pipeline(
        db_path=args.db,
        output_dir=args.output
    )

    results = {}

    # Step 1 & 2: YOLO training
    if not args.skip_yolo:
        data_yaml, prep_results = pipeline.step1_prepare_yolo_dataset()
        train_results, val_results = pipeline.step2_finetune_yolo(
            data_yaml=data_yaml,
            epochs=args.yolo_epochs,
            batch_size=args.batch_size
        )
        results['yolo'] = {
            'prep_results': prep_results,
            'train_results': train_results,
            'val_results': val_results
        }
    else:
        print("\n⏭️  Skipping YOLO training")
        results['yolo'] = {'skipped': True}

    # Step 3: Style classification
    if not args.skip_style:
        ensemble, individual_results, ensemble_acc = pipeline.step3_train_improved_style_classifier(
            epochs=args.style_epochs,
            batch_size=args.batch_size
        )
        results['style'] = {
            'ensemble': ensemble,
            'individual_results': individual_results,
            'ensemble_acc': ensemble_acc
        }
    else:
        print("\n⏭️  Skipping style classification training")
        results['style'] = {'skipped': True}

    # Generate final report
    if not args.skip_yolo or not args.skip_style:
        report = pipeline.generate_report(
            yolo_results=results.get('yolo', {}),
            style_results=results.get('style', {})
        )

    print("\n" + "=" * 80)
    print("✅ PHASE 2 PIPELINE COMPLETE!")
    print("=" * 80)
    print(f"\n📁 All outputs saved to: {args.output}")
    print("\n🎉 Your PlaybookTV Interior Design AI is now upgraded to Phase 2!")
    print("=" * 80)


if __name__ == "__main__":
    main()
