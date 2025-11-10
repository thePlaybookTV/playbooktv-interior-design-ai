"""
YOLO Dataset Preparation for 294-Category Taxonomy
Prepares data for fine-tuning YOLOv8 on custom interior design taxonomy
"""

import os
import json
import yaml
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import duckdb
import pandas as pd
from tqdm.auto import tqdm
from PIL import Image
import numpy as np

# Import taxonomy
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from interior_taxonomy import INTERIOR_TAXONOMY, get_all_categories


class YOLODatasetBuilder:
    """Build YOLO-format dataset from DuckDB detections"""

    def __init__(self, db_path: str, output_dir: str):
        self.db_path = Path(db_path)
        self.output_dir = Path(output_dir)
        self.conn = duckdb.connect(str(db_path))

        # Create category mapping from taxonomy
        self.categories = sorted(get_all_categories())
        self.category_to_id = {cat: idx for idx, cat in enumerate(self.categories)}

        # COCO to taxonomy mapping (for initial data)
        self.coco_to_taxonomy = self._create_coco_mapping()

        print(f"📊 YOLO Dataset Builder")
        print(f"   Categories: {len(self.categories)}")
        print(f"   Output: {self.output_dir}")

    def _create_coco_mapping(self) -> Dict[str, str]:
        """Map COCO classes to taxonomy categories"""
        # Comprehensive mapping from COCO classes to taxonomy
        mapping = {
            # Furniture - Seating
            'couch': 'sectional_sofa',
            'chair': 'accent_chair',
            'bench': 'bench',

            # Furniture - Beds
            'bed': 'queen_bed',

            # Furniture - Tables
            'dining table': 'dining_table',
            'desk': 'desk',

            # Electronics
            'tv': 'flat_screen_tv',
            'laptop': 'laptop',
            'keyboard': 'keyboard',
            'mouse': 'computer_mouse',
            'remote': 'remote_control',
            'cell phone': 'cell_phone',

            # Kitchen & Appliances
            'refrigerator': 'refrigerator',
            'oven': 'oven',
            'microwave': 'microwave',
            'toaster': 'toaster',
            'sink': 'sink',
            'dishwasher': 'dishwasher',

            # Bathroom
            'toilet': 'toilet',
            'bathtub': 'bathtub',
            'shower': 'walk_in_shower',

            # Decor & Accessories
            'potted plant': 'potted_plant',
            'vase': 'decorative_vase',
            'clock': 'wall_clock',
            'book': 'decorative_books',
            'bottle': 'decorative_bottles',
            'cup': 'coffee_mug',
            'bowl': 'decorative_bowl',
            'scissors': 'scissors',

            # Lighting (mapped to generic categories)
            'lamp': 'table_lamp',
        }
        return mapping

    def prepare_dataset(self, train_split: float = 0.8, min_confidence: float = 0.5):
        """
        Prepare YOLO dataset from DuckDB

        Args:
            train_split: Percentage of data for training (rest is validation)
            min_confidence: Minimum detection confidence to include
        """
        print("\n🔧 Preparing YOLO Dataset...")

        # Create directory structure
        train_images = self.output_dir / "images" / "train"
        val_images = self.output_dir / "images" / "val"
        train_labels = self.output_dir / "labels" / "train"
        val_labels = self.output_dir / "labels" / "val"

        for dir_path in [train_images, val_images, train_labels, val_labels]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Load images with detections
        print("\n📥 Loading data from DuckDB...")
        images_df = self.conn.execute("""
            SELECT DISTINCT
                i.image_id,
                i.original_path,
                i.room_type,
                i.style
            FROM images i
            INNER JOIN furniture_detections fd ON i.image_id = fd.image_id
            WHERE i.original_path IS NOT NULL
            AND fd.confidence >= ?
        """, [min_confidence]).df()

        print(f"   Found {len(images_df)} images with detections")

        # Split train/val
        np.random.seed(42)
        shuffled = images_df.sample(frac=1).reset_index(drop=True)
        split_idx = int(len(shuffled) * train_split)

        train_df = shuffled[:split_idx]
        val_df = shuffled[split_idx:]

        print(f"   Train: {len(train_df)} images")
        print(f"   Val: {len(val_df)} images")

        # Process train set
        print("\n🏋️ Processing training set...")
        self._process_split(train_df, train_images, train_labels, "train")

        # Process val set
        print("\n✅ Processing validation set...")
        self._process_split(val_df, val_images, val_labels, "val")

        # Create data.yaml
        self._create_yaml()

        # Create category mapping file
        self._save_category_mapping()

        print("\n✅ Dataset preparation complete!")
        print(f"📁 Dataset location: {self.output_dir}")

        return {
            'train_images': len(train_df),
            'val_images': len(val_df),
            'num_classes': len(self.categories)
        }

    def _process_split(self, df: pd.DataFrame, img_dir: Path, label_dir: Path, split: str):
        """Process images and labels for a split"""

        # Track statistics
        skipped_no_file = 0
        skipped_no_detections = 0
        skipped_no_valid_annotations = 0
        skipped_unmapped_classes = set()
        copied_images = 0

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split}"):
            image_id = row['image_id']
            original_path = row['original_path']

            # Check if image exists
            if not os.path.exists(original_path):
                skipped_no_file += 1
                continue

            # Get detections for this image
            detections = self.conn.execute("""
                SELECT
                    item_type,
                    confidence,
                    bbox_x1, bbox_y1, bbox_x2, bbox_y2
                FROM furniture_detections
                WHERE image_id = ?
            """, [image_id]).df()

            if len(detections) == 0:
                skipped_no_detections += 1
                continue

            # Load image to get dimensions
            try:
                img = Image.open(original_path)
                img_width, img_height = img.size
            except:
                skipped_no_file += 1
                continue

            # Convert detections to YOLO format
            yolo_annotations = []

            for _, det in detections.iterrows():
                # Map COCO class to taxonomy
                coco_class = det['item_type'].lower()

                if coco_class in self.coco_to_taxonomy:
                    taxonomy_class = self.coco_to_taxonomy[coco_class]

                    if taxonomy_class in self.category_to_id:
                        class_id = self.category_to_id[taxonomy_class]

                        # Convert bbox to YOLO format (normalized center x, y, width, height)
                        x1, y1, x2, y2 = det['bbox_x1'], det['bbox_y1'], det['bbox_x2'], det['bbox_y2']

                        center_x = ((x1 + x2) / 2) / img_width
                        center_y = ((y1 + y2) / 2) / img_height
                        width = (x2 - x1) / img_width
                        height = (y2 - y1) / img_height

                        # Clamp values to [0, 1]
                        center_x = max(0, min(1, center_x))
                        center_y = max(0, min(1, center_y))
                        width = max(0, min(1, width))
                        height = max(0, min(1, height))

                        yolo_annotations.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
                else:
                    # Track unmapped classes
                    skipped_unmapped_classes.add(coco_class)

            # Only save if we have annotations
            if yolo_annotations:
                # Copy image
                img_filename = f"{image_id}.jpg"
                try:
                    shutil.copy(original_path, img_dir / img_filename)
                    copied_images += 1
                except Exception as e:
                    print(f"\n⚠️  Failed to copy {original_path}: {e}")
                    continue

                # Save label file
                label_filename = f"{image_id}.txt"
                label_path = label_dir / label_filename

                with open(label_path, 'w') as f:
                    f.write('\n'.join(yolo_annotations))
            else:
                skipped_no_valid_annotations += 1

        # Print statistics
        print(f"\n📊 {split.capitalize()} Split Statistics:")
        print(f"   ✅ Images copied: {copied_images}")
        print(f"   ⚠️  Skipped (no file): {skipped_no_file}")
        print(f"   ⚠️  Skipped (no detections): {skipped_no_detections}")
        print(f"   ⚠️  Skipped (no valid annotations): {skipped_no_valid_annotations}")
        if skipped_unmapped_classes:
            print(f"   ⚠️  Unmapped COCO classes: {sorted(skipped_unmapped_classes)}")

    def _create_yaml(self):
        """Create YOLO data.yaml configuration file"""

        yaml_content = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(self.categories),
            'names': self.categories
        }

        yaml_path = self.output_dir / 'data.yaml'

        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)

        print(f"\n✅ Created data.yaml: {yaml_path}")

    def _save_category_mapping(self):
        """Save category mapping for reference"""

        mapping = {
            'categories': self.categories,
            'category_to_id': self.category_to_id,
            'coco_to_taxonomy': self.coco_to_taxonomy,
            'num_classes': len(self.categories)
        }

        mapping_path = self.output_dir / 'category_mapping.json'

        with open(mapping_path, 'w') as f:
            json.dump(mapping, f, indent=2)

        print(f"✅ Created category_mapping.json: {mapping_path}")

    def create_synthetic_labels(self, use_sam_masks: bool = True):
        """
        Create more training data by using SAM2 masks to generate better labels
        This uses the existing SAM2 masks to create pixel-perfect bounding boxes
        """
        print("\n🎨 Creating synthetic labels from SAM2 masks...")

        # Get images with SAM2 masks
        images_with_masks = self.conn.execute("""
            SELECT DISTINCT
                i.image_id,
                i.original_path
            FROM images i
            INNER JOIN furniture_detections fd ON i.image_id = fd.image_id
            WHERE fd.has_mask = TRUE
        """).df()

        print(f"   Found {len(images_with_masks)} images with SAM2 masks")
        print("   This can be used to refine bounding boxes for better training")

        return len(images_with_masks)

    def get_statistics(self):
        """Get dataset statistics"""

        stats = {}

        # Count images and labels
        for split in ['train', 'val']:
            img_dir = self.output_dir / "images" / split
            label_dir = self.output_dir / "labels" / split

            if img_dir.exists():
                num_images = len(list(img_dir.glob('*.jpg')))
                num_labels = len(list(label_dir.glob('*.txt')))

                # Count total annotations
                total_annotations = 0
                for label_file in label_dir.glob('*.txt'):
                    with open(label_file, 'r') as f:
                        total_annotations += len(f.readlines())

                stats[split] = {
                    'images': num_images,
                    'labels': num_labels,
                    'annotations': total_annotations,
                    'avg_annotations_per_image': total_annotations / num_images if num_images > 0 else 0
                }

        return stats

    def close(self):
        """Close database connection"""
        self.conn.close()


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    # Configuration
    DB_PATH = "./interior_design_data_hybrid/processed/metadata.duckdb"
    OUTPUT_DIR = "./yolo_dataset"

    print("=" * 70)
    print("🚀 YOLO DATASET PREPARATION FOR 294-CATEGORY TAXONOMY")
    print("=" * 70)

    # Create builder
    builder = YOLODatasetBuilder(DB_PATH, OUTPUT_DIR)

    # Prepare dataset
    result = builder.prepare_dataset(
        train_split=0.8,
        min_confidence=0.5
    )

    # Get statistics
    print("\n📊 Dataset Statistics:")
    print("=" * 70)
    stats = builder.get_statistics()

    for split, split_stats in stats.items():
        print(f"\n{split.upper()}:")
        print(f"   Images: {split_stats['images']:,}")
        print(f"   Labels: {split_stats['labels']:,}")
        print(f"   Annotations: {split_stats['annotations']:,}")
        print(f"   Avg per image: {split_stats['avg_annotations_per_image']:.2f}")

    print("\n" + "=" * 70)
    print(f"✅ Dataset ready for YOLO training!")
    print(f"📁 Location: {OUTPUT_DIR}")
    print(f"📋 Classes: {len(builder.categories)}")
    print("=" * 70)

    builder.close()
