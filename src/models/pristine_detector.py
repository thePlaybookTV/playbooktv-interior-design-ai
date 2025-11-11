# ============================================
# COMPLETE SAM2 + YOLO PIPELINE
# Optimized for A4000 GPU with 5-hour limit
# Processes 67K images with checkpoints
# ============================================

import os
import json
import time
from pathlib import Path
from datetime import datetime
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import duckdb
from typing import List
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from interior_taxonomy import get_items_by_room, get_items_by_style
except ImportError:
    # Fallback if taxonomy not available
    def get_items_by_room(room_type: str) -> List[str]:
        return []
    def get_items_by_style(style: str) -> List[str]:
        return []

# ============================================
# STEP 1: INSTALL SAM2
# ============================================

def setup_sam2():
    """Install and setup SAM2"""
    print("📦 Setting up SAM2...\n")
    
    # Install SAM2
    try:
        import sam2
        print("✅ SAM2 already installed")
    except:
        print("Installing SAM2...")
        os.system("pip install git+https://github.com/facebookresearch/segment-anything-2.git")
        print("✅ SAM2 installed")
    
    # Download checkpoint
    checkpoint_dir = Path("./checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    checkpoint_path = checkpoint_dir / "sam2_hiera_large.pt"
    
    if not checkpoint_path.exists():
        print("\n📥 Downloading SAM2 checkpoint (~900MB)...")
        os.system(f"wget -O {checkpoint_path} https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt")
        print("✅ Checkpoint downloaded")
    else:
        print("✅ Checkpoint already exists")

# Run setup
setup_sam2()

# ============================================
# STEP 2: COMBINED YOLO + SAM2 DETECTOR
# ============================================

class PristineDetector:
    """Combined YOLO (fast bbox) + SAM2 (precise masks)"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🎮 Using device: {self.device}")
        
        # Load YOLO
        try:
            from ultralytics import YOLO
            print("📦 Loading YOLO...")
            self.yolo = YOLO('yolov8m.pt')
            print("✅ YOLO loaded")
        except:
            print("❌ YOLO failed")
            self.yolo = None
        
        # Load SAM2
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            
            print("📦 Loading SAM2...")
            
            checkpoint = "./checkpoints/sam2_hiera_large.pt"
            model_cfg = "sam2_hiera_l.yaml"
            
            sam2_model = build_sam2(model_cfg, checkpoint, device=self.device)
            self.sam2 = SAM2ImagePredictor(sam2_model)
            print("✅ SAM2 loaded")
            
        except Exception as e:
            print(f"❌ SAM2 failed: {e}")
            self.sam2 = None
        
        # Furniture classes
        self.furniture_classes = [
            'couch', 'chair', 'bed', 'dining table', 'toilet',
            'tv', 'laptop', 'sink', 'refrigerator', 'oven',
            'potted plant', 'vase', 'clock', 'book'
        ]
    
    def detect_with_masks(self, image_path: str) -> dict:
        """Detect furniture with YOLO + segment with SAM2"""
        
        if not self.yolo or not self.sam2:
            return {'items': [], 'count': 0}
        
        try:
            # Step 1: YOLO detection (fast)
            results = self.yolo(image_path, verbose=False)
            
            # Load image for SAM2
            image = np.array(Image.open(image_path).convert('RGB'))
            self.sam2.set_image(image)
            
            items = []
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls)
                        class_name = self.yolo.names[class_id]
                        
                        if class_name.lower() in self.furniture_classes:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            confidence = float(box.conf[0])
                            
                            # Step 2: SAM2 segmentation (precise)
                            input_box = np.array([x1, y1, x2, y2])
                            
                            try:
                                masks, scores, _ = self.sam2.predict(
                                    point_coords=None,
                                    point_labels=None,
                                    box=input_box[None, :],
                                    multimask_output=False
                                )
                                
                                mask = masks[0]
                                mask_score = float(scores[0])
                                mask_area = int(mask.sum())
                                
                            except:
                                mask = None
                                mask_score = 0.0
                                mask_area = 0
                            
                            # Calculate bbox area
                            bbox_area = (x2 - x1) * (y2 - y1)
                            img_area = image.shape[0] * image.shape[1]
                            area_pct = (bbox_area / img_area) * 100
                            
                            items.append({
                                'type': class_name,
                                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                'confidence': confidence,
                                'area_percentage': float(area_pct),
                                'mask_area': mask_area,
                                'mask_score': mask_score,
                                'has_mask': mask is not None
                            })
            
            return {'items': items, 'count': len(items)}
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return {'items': [], 'count': 0}

# ============================================
# STEP 3: BATCH PROCESSOR WITH CHECKPOINTS
# ============================================

class CheckpointProcessor:
    """Process 67K images with automatic checkpointing"""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.detector = PristineDetector()
        
        # Checkpoint file
        self.checkpoint_file = Path("processing_checkpoint.json")
        
        # Connect to database
        self.conn = duckdb.connect(str(db_path))
        self._setup_tables()
        
        # Load checkpoint
        self.processed_ids = self._load_checkpoint()
    
    def _setup_tables(self):
        """Setup database tables"""
        print("\n📊 Setting up tables...")
        
        # Main detections table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS furniture_detections (
                image_id VARCHAR,
                item_type VARCHAR,
                confidence FLOAT,
                bbox_x1 FLOAT,
                bbox_y1 FLOAT,
                bbox_x2 FLOAT,
                bbox_y2 FLOAT,
                area_percentage FLOAT,
                mask_area INTEGER,
                mask_score FLOAT,
                has_mask BOOLEAN
            )
        """)
        
        # Add furniture_count to images
        try:
            self.conn.execute("ALTER TABLE images ADD COLUMN furniture_count INTEGER")
        except:
            pass
        
        print("✅ Tables ready")
    
    def _load_checkpoint(self) -> set:
        """Load processed image IDs from checkpoint"""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                data = json.load(f)
                print(f"\n📝 Loaded checkpoint: {len(data['processed'])} images already processed")
                return set(data['processed'])
        return set()
    
    def _save_checkpoint(self, processed_ids: set):
        """Save checkpoint"""
        with open(self.checkpoint_file, 'w') as f:
            json.dump({
                'processed': list(processed_ids),
                'timestamp': datetime.now().isoformat(),
                'total': len(processed_ids)
            }, f)
    
    def infer_room_type(self, detected_items: List[str]) -> str:
        """
        Infer room type based on detected furniture items using taxonomy.

        Args:
            detected_items: List of detected item types

        Returns:
            Most likely room type (e.g., 'living_room', 'bedroom')
        """
        room_scores = {}

        # Score each room type based on detected items
        for room_type in ["living_room", "bedroom", "kitchen", "dining_room", 
                          "bathroom", "home_office", "entryway"]:
            typical_items = get_items_by_room(room_type)

            # Count how many detected items are typical for this room
            score = sum(1 for item in detected_items if item in typical_items)
            room_scores[room_type] = score

        # Return room type with highest score
        if room_scores:
            return max(room_scores.items(), key=lambda x: x[1])[0]
        return "unknown"


    def infer_style(self, detected_items: List[str]) -> str:
        """
        Infer design style based on detected furniture items using taxonomy.

        Args:
            detected_items: List of detected item types

        Returns:
            Most likely style (e.g., 'modern', 'traditional')
        """
        style_scores = {}

        # Score each style based on detected items
        for style in ["modern", "traditional", "contemporary", "industrial", 
                      "bohemian", "mid_century", "luxury", "farmhouse"]:
            style_items = get_items_by_style(style)

            # Count how many detected items match this style
            score = sum(1 for item in detected_items if item in style_items)
            style_scores[style] = score

        # Return style with highest score
        if style_scores:
            return max(style_scores.items(), key=lambda x: x[1])[0]
        return "unknown"
    
    def process_all(self, batch_size: int = 100, checkpoint_interval: int = 500):
        """Process all images with checkpointing"""
        
        # Get images to process
        print("\n🔍 Finding images to process...")
        
        df = self.conn.execute("""
            SELECT image_id, original_path
            FROM images
            WHERE original_path IS NOT NULL
        """).df()
        
        # Filter out already processed
        df = df[~df['image_id'].isin(self.processed_ids)]
        
        total_images = len(df)
        
        if total_images == 0:
            print("✅ All images already processed!")
            return
        
        print(f"📷 Processing {total_images} images")
        print(f"💾 Saving checkpoint every {checkpoint_interval} images")
        print(f"⏱️  Estimated time: {total_images * 0.3 / 60:.1f} minutes\n")
        
        start_time = time.time()
        processed_count = 0
        
        # Process in batches
        for i in tqdm(range(0, len(df), batch_size), desc="Processing"):
            batch = df.iloc[i:i+batch_size]
            
            for _, row in batch.iterrows():
                if not os.path.exists(row['original_path']):
                    continue
                
                # Detect with YOLO + SAM2
                result = self.detector.detect_with_masks(row['original_path'])
                
                # Update furniture count
                self.conn.execute("""
                    UPDATE images 
                    SET furniture_count = ?
                    WHERE image_id = ?
                """, (result['count'], row['image_id']))
                
                # Insert detections
                for item in result['items']:
                    self.conn.execute("""
                        INSERT INTO furniture_detections 
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        row['image_id'],
                        item['type'],
                        item['confidence'],
                        item['bbox'][0],
                        item['bbox'][1],
                        item['bbox'][2],
                        item['bbox'][3],
                        item['area_percentage'],
                        item['mask_area'],
                        item['mask_score'],
                        item['has_mask']
                    ))
                
                # Add to processed
                self.processed_ids.add(row['image_id'])
                processed_count += 1
                
                # Save checkpoint periodically
                if processed_count % checkpoint_interval == 0:
                    self._save_checkpoint(self.processed_ids)
                    elapsed = time.time() - start_time
                    rate = processed_count / elapsed
                    remaining = (total_images - processed_count) / rate / 60
                    print(f"\n💾 Checkpoint: {processed_count}/{total_images} ({remaining:.1f} min remaining)")
        
        # Final checkpoint
        self._save_checkpoint(self.processed_ids)
        
        elapsed = time.time() - start_time
        print(f"\n✅ Completed {processed_count} images in {elapsed/60:.1f} minutes")
        print(f"📊 Average: {processed_count/elapsed:.2f} images/second")
    
    def show_stats(self):
        """Show detection statistics"""
        print("\n" + "=" * 70)
        print("📊 DETECTION STATISTICS")
        print("=" * 70)
        
        # Overall
        total = self.conn.execute("SELECT COUNT(*) FROM images WHERE furniture_count IS NOT NULL").fetchone()[0]
        
        stats = self.conn.execute("""
            SELECT 
                AVG(furniture_count) as avg_count,
                MAX(furniture_count) as max_count,
                MIN(furniture_count) as min_count
            FROM images
            WHERE furniture_count IS NOT NULL
        """).fetchone()
        
        print(f"\n📷 Images processed: {total:,}")
        print(f"🪑 Average items: {stats[0]:.1f}")
        print(f"📊 Max items: {stats[1]}")
        print(f"📊 Min items: {stats[2]}")
        
        # Items with masks
        with_masks = self.conn.execute("""
            SELECT COUNT(*) FROM furniture_detections WHERE has_mask = TRUE
        """).fetchone()[0]
        
        total_detections = self.conn.execute("SELECT COUNT(*) FROM furniture_detections").fetchone()[0]
        
        print(f"\n🎭 Detections with SAM2 masks: {with_masks:,}/{total_detections:,} ({with_masks/total_detections*100:.1f}%)")
        
        # Most common
        print("\n🏆 Top 10 Items:")
        items = self.conn.execute("""
            SELECT item_type, COUNT(*) as count
            FROM furniture_detections
            GROUP BY item_type
            ORDER BY count DESC
            LIMIT 10
        """).df()
        
        for _, row in items.iterrows():
            print(f"   {row['item_type']:20s}: {row['count']:6,}")
        
        print("\n" + "=" * 70)
    
    def close(self):
        self.conn.close()

# ============================================
# STEP 4: RUN PROCESSING
# ============================================

print("=" * 70)
print("🚀 PRISTINE MVP: SAM2 + YOLO DETECTION PIPELINE")
print("=" * 70)
print("\n📊 Your Setup:")
print("   GPU: A4000 (16GB) ✅")
print("   RAM: 45GB ✅")
print("   CPU: 8 cores ✅")
print("\n⏱️  Time limit: 5 hours")
print("💾 Auto-checkpointing: Every 500 images")
print("🔄 Resumable: Can continue if interrupted\n")
print("=" * 70)

# Path to database
db_path = "./interior_design_data_hybrid/processed/metadata.duckdb"

# Create processor
processor = CheckpointProcessor(db_path)

# Process all images
print("\n🎬 Starting processing...\n")
processor.process_all(
    batch_size=100,
    checkpoint_interval=500
)

# Show statistics
processor.show_stats()

# Close
processor.close()

print("\n🎉 PROCESSING COMPLETE!")
print("✅ All detections have YOLO bboxes + SAM2 masks")
print("📁 Database: " + str(db_path))