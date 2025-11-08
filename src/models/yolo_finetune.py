"""
YOLO Fine-Tuning Script for 294-Category Interior Design Taxonomy
Fine-tunes YOLOv8 on custom dataset with advanced features
"""

import os
from pathlib import Path
from datetime import datetime
import torch
from ultralytics import YOLO
import yaml
import json
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import pandas as pd


class YOLOFineTuner:
    """Fine-tune YOLO on custom interior design taxonomy"""

    def __init__(self, data_yaml: str, model_size: str = 'yolov8m.pt'):
        """
        Initialize YOLO fine-tuner

        Args:
            data_yaml: Path to YOLO data.yaml file
            model_size: Base model to use (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
        """
        self.data_yaml = Path(data_yaml)
        self.model_size = model_size

        # Check if GPU is available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🎮 Using device: {self.device}")

        # Load data config
        with open(data_yaml, 'r') as f:
            self.data_config = yaml.safe_load(f)

        self.num_classes = self.data_config['nc']
        print(f"📊 Number of classes: {self.num_classes}")

        # Create output directory
        self.output_dir = Path("./yolo_training_runs")
        self.output_dir.mkdir(exist_ok=True)

    def train(
        self,
        epochs: int = 100,
        batch_size: int = 16,
        image_size: int = 640,
        learning_rate: float = 0.01,
        patience: int = 50,
        save_period: int = 10,
        augment: bool = True,
        freeze_layers: int = 0,
        resume: bool = False
    ):
        """
        Train YOLO model

        Args:
            epochs: Number of training epochs
            batch_size: Batch size (adjust based on GPU memory)
            image_size: Input image size
            learning_rate: Initial learning rate
            patience: Early stopping patience
            save_period: Save checkpoint every N epochs
            augment: Use data augmentation
            freeze_layers: Number of layers to freeze (0 = train all)
            resume: Resume from last checkpoint
        """

        print("\n" + "=" * 70)
        print(f"🚀 STARTING YOLO FINE-TUNING")
        print("=" * 70)
        print(f"\nTraining Configuration:")
        print(f"   Model: {self.model_size}")
        print(f"   Classes: {self.num_classes}")
        print(f"   Epochs: {epochs}")
        print(f"   Batch Size: {batch_size}")
        print(f"   Image Size: {image_size}")
        print(f"   Learning Rate: {learning_rate}")
        print(f"   Device: {self.device}")
        print(f"   Augmentation: {augment}")
        print(f"   Freeze Layers: {freeze_layers}")
        print("=" * 70 + "\n")

        # Load model
        print(f"📦 Loading {self.model_size}...")
        model = YOLO(self.model_size)

        # Train
        print(f"\n🏋️ Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        results = model.train(
            data=str(self.data_yaml),
            epochs=epochs,
            batch=batch_size,
            imgsz=image_size,
            lr0=learning_rate,
            patience=patience,
            save_period=save_period,
            device=self.device,
            augment=augment,
            freeze=freeze_layers,
            resume=resume,
            project=str(self.output_dir),
            name='finetune_294_classes',
            exist_ok=True,
            pretrained=True,
            optimizer='AdamW',
            verbose=True,
            # Advanced settings
            cos_lr=True,  # Cosine learning rate scheduler
            close_mosaic=10,  # Disable mosaic augmentation in last N epochs
            amp=True,  # Automatic mixed precision
            dropout=0.0,  # Dropout (0.0 to disable)
            weight_decay=0.0005,  # Weight decay
            warmup_epochs=3.0,  # Warmup epochs
            warmup_momentum=0.8,  # Warmup momentum
            warmup_bias_lr=0.1,  # Warmup bias learning rate
            box=7.5,  # Box loss gain
            cls=0.5,  # Class loss gain
            dfl=1.5,  # DFL loss gain
            mosaic=1.0,  # Mosaic augmentation probability
            mixup=0.0,  # Mixup augmentation probability
            copy_paste=0.0,  # Copy-paste augmentation probability
            degrees=0.0,  # Rotation degrees
            translate=0.1,  # Translation
            scale=0.5,  # Scale
            shear=0.0,  # Shear
            perspective=0.0,  # Perspective
            flipud=0.0,  # Flip up-down probability
            fliplr=0.5,  # Flip left-right probability
            hsv_h=0.015,  # HSV hue augmentation
            hsv_s=0.7,  # HSV saturation augmentation
            hsv_v=0.4,  # HSV value augmentation
        )

        print(f"\n✅ Training completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Save training summary
        self._save_training_summary(results)

        return results

    def validate(self, weights_path: str = None):
        """
        Validate model on validation set

        Args:
            weights_path: Path to model weights (uses best.pt if None)
        """

        if weights_path is None:
            # Use best weights from latest training
            latest_run = self.output_dir / 'finetune_294_classes'
            weights_path = latest_run / 'weights' / 'best.pt'

        print(f"\n📊 Validating model: {weights_path}")

        model = YOLO(str(weights_path))

        results = model.val(
            data=str(self.data_yaml),
            device=self.device,
            verbose=True
        )

        print("\n✅ Validation Results:")
        print(f"   mAP50: {results.box.map50:.4f}")
        print(f"   mAP50-95: {results.box.map:.4f}")
        print(f"   Precision: {results.box.mp:.4f}")
        print(f"   Recall: {results.box.mr:.4f}")

        return results

    def export_model(self, weights_path: str = None, formats: list = ['onnx', 'torchscript']):
        """
        Export model to different formats

        Args:
            weights_path: Path to model weights
            formats: List of export formats (onnx, torchscript, tflite, etc.)
        """

        if weights_path is None:
            latest_run = self.output_dir / 'finetune_294_classes'
            weights_path = latest_run / 'weights' / 'best.pt'

        print(f"\n📦 Exporting model: {weights_path}")

        model = YOLO(str(weights_path))

        for fmt in formats:
            print(f"\n   Exporting to {fmt}...")
            try:
                model.export(format=fmt)
                print(f"   ✅ {fmt} export complete")
            except Exception as e:
                print(f"   ❌ {fmt} export failed: {e}")

    def _save_training_summary(self, results):
        """Save training summary to JSON"""

        latest_run = self.output_dir / 'finetune_294_classes'

        summary = {
            'timestamp': datetime.now().isoformat(),
            'model_size': self.model_size,
            'num_classes': self.num_classes,
            'device': self.device,
            'data_yaml': str(self.data_yaml),
        }

        summary_path = latest_run / 'training_summary.json'

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n✅ Training summary saved: {summary_path}")

    def test_inference(self, image_path: str, weights_path: str = None, conf_threshold: float = 0.25):
        """
        Test inference on a single image

        Args:
            image_path: Path to test image
            weights_path: Path to model weights
            conf_threshold: Confidence threshold
        """

        if weights_path is None:
            latest_run = self.output_dir / 'finetune_294_classes'
            weights_path = latest_run / 'weights' / 'best.pt'

        print(f"\n🔍 Testing inference on: {image_path}")

        model = YOLO(str(weights_path))

        results = model.predict(
            image_path,
            conf=conf_threshold,
            device=self.device,
            verbose=True
        )

        # Display results
        for r in results:
            print(f"\n   Detections: {len(r.boxes)}")

            for box in r.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = self.data_config['names'][class_id]

                print(f"   - {class_name}: {confidence:.3f}")

            # Save annotated image
            annotated = r.plot()
            output_path = Path(image_path).parent / f"annotated_{Path(image_path).name}"

            from PIL import Image
            Image.fromarray(annotated).save(output_path)

            print(f"\n   💾 Annotated image saved: {output_path}")

        return results


class YOLOTrainingMonitor:
    """Monitor and visualize YOLO training progress"""

    def __init__(self, results_csv: str):
        """
        Initialize monitor

        Args:
            results_csv: Path to results.csv from YOLO training
        """
        self.results_csv = Path(results_csv)

        if self.results_csv.exists():
            self.df = pd.read_csv(results_csv)
            print(f"✅ Loaded training results: {len(self.df)} epochs")
        else:
            print(f"⚠️ Results file not found: {results_csv}")
            self.df = None

    def plot_metrics(self, save_path: str = None):
        """Plot training metrics"""

        if self.df is None:
            print("❌ No data to plot")
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('YOLO Training Metrics', fontsize=16, fontweight='bold')

        # Loss curves
        if 'train/box_loss' in self.df.columns:
            axes[0, 0].plot(self.df['train/box_loss'], label='Train', linewidth=2)
            if 'val/box_loss' in self.df.columns:
                axes[0, 0].plot(self.df['val/box_loss'], label='Val', linewidth=2)
            axes[0, 0].set_title('Box Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        if 'train/cls_loss' in self.df.columns:
            axes[0, 1].plot(self.df['train/cls_loss'], label='Train', linewidth=2)
            if 'val/cls_loss' in self.df.columns:
                axes[0, 1].plot(self.df['val/cls_loss'], label='Val', linewidth=2)
            axes[0, 1].set_title('Classification Loss')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        if 'train/dfl_loss' in self.df.columns:
            axes[0, 2].plot(self.df['train/dfl_loss'], label='Train', linewidth=2)
            if 'val/dfl_loss' in self.df.columns:
                axes[0, 2].plot(self.df['val/dfl_loss'], label='Val', linewidth=2)
            axes[0, 2].set_title('DFL Loss')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Loss')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)

        # Metrics
        if 'metrics/precision(B)' in self.df.columns:
            axes[1, 0].plot(self.df['metrics/precision(B)'], linewidth=2, color='green')
            axes[1, 0].set_title('Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].grid(True, alpha=0.3)

        if 'metrics/recall(B)' in self.df.columns:
            axes[1, 1].plot(self.df['metrics/recall(B)'], linewidth=2, color='orange')
            axes[1, 1].set_title('Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].grid(True, alpha=0.3)

        if 'metrics/mAP50(B)' in self.df.columns:
            axes[1, 2].plot(self.df['metrics/mAP50(B)'], linewidth=2, color='blue', label='mAP50')
            if 'metrics/mAP50-95(B)' in self.df.columns:
                axes[1, 2].plot(self.df['metrics/mAP50-95(B)'], linewidth=2, color='red', label='mAP50-95')
            axes[1, 2].set_title('mAP Metrics')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('mAP')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Training plots saved: {save_path}")

        plt.show()


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Fine-tune YOLO on 294-category taxonomy')
    parser.add_argument('--data', type=str, default='./yolo_dataset/data.yaml',
                        help='Path to data.yaml')
    parser.add_argument('--model', type=str, default='yolov8m.pt',
                        help='Base model (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'validate', 'export', 'test'],
                        help='Operation mode')
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to weights for validation/export/test')
    parser.add_argument('--test-image', type=str, default=None,
                        help='Path to test image')

    args = parser.parse_args()

    # Create fine-tuner
    finetuner = YOLOFineTuner(
        data_yaml=args.data,
        model_size=args.model
    )

    if args.mode == 'train':
        # Train model
        results = finetuner.train(
            epochs=args.epochs,
            batch_size=args.batch,
            image_size=args.imgsz,
            learning_rate=args.lr,
            patience=50,
            save_period=10,
            augment=True,
            freeze_layers=0
        )

        # Validate after training
        print("\n" + "=" * 70)
        print("📊 VALIDATION ON BEST MODEL")
        print("=" * 70)
        finetuner.validate()

    elif args.mode == 'validate':
        # Validate model
        finetuner.validate(weights_path=args.weights)

    elif args.mode == 'export':
        # Export model
        finetuner.export_model(
            weights_path=args.weights,
            formats=['onnx', 'torchscript']
        )

    elif args.mode == 'test':
        # Test inference
        if args.test_image:
            finetuner.test_inference(
                image_path=args.test_image,
                weights_path=args.weights
            )
        else:
            print("❌ Please provide --test-image path")

    print("\n" + "=" * 70)
    print("✅ YOLO FINE-TUNING COMPLETE!")
    print("=" * 70)
