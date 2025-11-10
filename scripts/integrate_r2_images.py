"""
Integrate Cloudflare R2 Images into Training Pipeline
Downloads images from R2 bucket and adds them to DuckDB for training
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb
import pandas as pd
from tqdm.auto import tqdm
from PIL import Image
import hashlib
import json
from datetime import datetime
from typing import Optional, Dict, List
import argparse

from src.data_collection.cloudflare_r2_downloader import CloudflareR2Downloader


class R2ImageIntegrator:
    """Integrate R2 images into training database"""

    def __init__(self, db_path: str, r2_downloader: CloudflareR2Downloader):
        self.db_path = Path(db_path)
        self.downloader = r2_downloader
        self.conn = duckdb.connect(str(db_path))

        print(f"📊 R2 Image Integrator")
        print(f"   Database: {self.db_path}")

    def download_and_process(
        self,
        output_dir: str | Path,
        prefix: str = "",
        max_images: Optional[int] = None,
        default_room_type: str = "living_room",
        default_style: str = "modern",
        run_detection: bool = True,
        metadata_file: Optional[str] = None
    ) -> Dict:
        """
        Download R2 images and process them for training

        Args:
            output_dir: Local directory to download images
            prefix: R2 bucket prefix to filter images
            max_images: Maximum number of images to download
            default_room_type: Default room type for images (if not in metadata)
            default_style: Default style for images (if not in metadata)
            run_detection: Whether to run furniture detection on downloaded images
            metadata_file: Optional JSON file with image metadata (room_type, style, etc.)

        Returns:
            Statistics dictionary
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load metadata if provided
        metadata = {}
        if metadata_file and Path(metadata_file).exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            print(f"📋 Loaded metadata for {len(metadata)} images")

        # Download images from R2
        print("\n" + "=" * 70)
        print("📥 STEP 1: DOWNLOADING FROM R2")
        print("=" * 70)

        downloaded = self.downloader.download_all_images(
            output_dir=output_dir,
            prefix=prefix,
            max_images=max_images,
            keep_structure=False,  # Flatten structure
            skip_existing=True
        )

        if not downloaded:
            print("⚠️  No images downloaded")
            return {'downloaded': 0, 'processed': 0, 'detected': 0}

        # Process downloaded images
        print("\n" + "=" * 70)
        print("🔍 STEP 2: PROCESSING IMAGES")
        print("=" * 70)

        processed_images = []

        for r2_key, local_path in tqdm(downloaded.items(), desc="Processing"):
            try:
                # Open and validate image
                img = Image.open(local_path)
                width, height = img.size

                # Skip invalid images
                if width < 224 or height < 224:
                    print(f"\n⚠️  Skipping {local_path.name}: too small ({width}x{height})")
                    continue

                # Generate image ID
                image_id = self._generate_image_id(local_path)

                # Get metadata
                img_metadata = metadata.get(r2_key, {})
                room_type = img_metadata.get('room_type', default_room_type)
                style = img_metadata.get('style', default_style)
                tags = img_metadata.get('tags', [])

                # Create image record
                image_record = {
                    'image_id': image_id,
                    'original_path': str(local_path.absolute()),
                    'r2_key': r2_key,
                    'room_type': room_type,
                    'style': style,
                    'width': width,
                    'height': height,
                    'source': 'cloudflare_r2',
                    'tags': json.dumps(tags),
                    'created_at': datetime.now().isoformat()
                }

                processed_images.append(image_record)

            except Exception as e:
                print(f"\n❌ Error processing {local_path}: {e}")
                continue

        print(f"\n✅ Processed {len(processed_images)} images")

        # Add to database
        print("\n" + "=" * 70)
        print("💾 STEP 3: ADDING TO DATABASE")
        print("=" * 70)

        self._add_images_to_db(processed_images)

        # Run detection if requested
        detections_count = 0
        if run_detection:
            print("\n" + "=" * 70)
            print("🔍 STEP 4: RUNNING FURNITURE DETECTION")
            print("=" * 70)
            detections_count = self._run_detection_on_images(processed_images)

        # Return statistics
        stats = {
            'downloaded': len(downloaded),
            'processed': len(processed_images),
            'detected': detections_count
        }

        print("\n" + "=" * 70)
        print("✅ INTEGRATION COMPLETE!")
        print("=" * 70)
        print(f"   Downloaded: {stats['downloaded']} images")
        print(f"   Processed: {stats['processed']} images")
        print(f"   Detections: {stats['detected']} objects")

        return stats

    def _generate_image_id(self, file_path: Path) -> str:
        """Generate unique image ID from file"""
        # Use file name and size for ID
        with open(file_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()[:12]
        return f"r2_{file_hash}"

    def _add_images_to_db(self, image_records: List[Dict]):
        """Add image records to database"""

        df = pd.DataFrame(image_records)

        print(f"\n📊 Adding {len(df)} images to database...")

        try:
            # Check if images already exist
            existing_ids = set()
            try:
                existing_df = self.conn.execute("""
                    SELECT image_id FROM images WHERE source = 'cloudflare_r2'
                """).df()
                existing_ids = set(existing_df['image_id'].tolist())
                print(f"   Found {len(existing_ids)} existing R2 images")
            except:
                # Table might not exist yet
                pass

            # Filter out existing images
            new_df = df[~df['image_id'].isin(existing_ids)]

            if len(new_df) == 0:
                print("   All images already exist in database")
                return

            print(f"   Inserting {len(new_df)} new images...")

            # Insert new images
            self.conn.execute("""
                INSERT INTO images (
                    image_id, original_path, room_type, style,
                    width, height, source
                )
                SELECT
                    image_id, original_path, room_type, style,
                    width, height, source
                FROM new_df
            """)

            print(f"   ✅ Added {len(new_df)} images to database")

        except Exception as e:
            print(f"❌ Error adding images to database: {e}")
            print("   Creating images table if it doesn't exist...")

            # Create table if needed
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS images (
                    image_id VARCHAR PRIMARY KEY,
                    original_path VARCHAR,
                    room_type VARCHAR,
                    style VARCHAR,
                    width INTEGER,
                    height INTEGER,
                    furniture_count INTEGER,
                    source VARCHAR,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Try insert again
            self.conn.execute("""
                INSERT INTO images (
                    image_id, original_path, room_type, style,
                    width, height, source
                )
                SELECT
                    image_id, original_path, room_type, style,
                    width, height, source
                FROM df
            """)

            print(f"   ✅ Added {len(df)} images to database")

    def _run_detection_on_images(self, image_records: List[Dict]) -> int:
        """Run furniture detection on images"""

        # Import detection models
        try:
            from ultralytics import YOLO
            print("\n🤖 Loading YOLO model for detection...")

            # Use YOLO11 or YOLOv8
            model = YOLO('yolo11x.pt')  # or 'yolov8x.pt'
            print("   ✅ Model loaded")

        except Exception as e:
            print(f"❌ Failed to load YOLO model: {e}")
            print("   Skipping detection step")
            print("   Install ultralytics: pip install ultralytics")
            return 0

        detection_records = []
        total_detections = 0

        print(f"\n🔍 Running detection on {len(image_records)} images...")

        for record in tqdm(image_records, desc="Detecting furniture"):
            try:
                image_path = record['original_path']
                image_id = record['image_id']

                # Run detection
                results = model(image_path, verbose=False)

                # Process detections
                for result in results:
                    boxes = result.boxes
                    for i in range(len(boxes)):
                        # Get box data
                        box = boxes[i]
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        xyxy = box.xyxy[0].tolist()

                        # Get class name
                        class_name = model.names[cls]

                        # Create detection record
                        detection_id = f"{image_id}_det_{i}"

                        detection_records.append({
                            'detection_id': detection_id,
                            'image_id': image_id,
                            'item_type': class_name,
                            'confidence': conf,
                            'bbox_x1': xyxy[0],
                            'bbox_y1': xyxy[1],
                            'bbox_x2': xyxy[2],
                            'bbox_y2': xyxy[3],
                            'area_percentage': (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1]) / (record['width'] * record['height']) * 100,
                            'source': 'yolo_auto_r2'
                        })

                        total_detections += 1

            except Exception as e:
                print(f"\n❌ Detection failed for {record['original_path']}: {e}")
                continue

        # Add detections to database
        if detection_records:
            print(f"\n💾 Adding {len(detection_records)} detections to database...")

            detections_df = pd.DataFrame(detection_records)

            try:
                self.conn.execute("""
                    INSERT INTO furniture_detections
                    SELECT * FROM detections_df
                """)
                print(f"   ✅ Added {len(detection_records)} detections")

                # Update furniture counts
                self.conn.execute("""
                    UPDATE images
                    SET furniture_count = (
                        SELECT COUNT(*)
                        FROM furniture_detections
                        WHERE furniture_detections.image_id = images.image_id
                    )
                    WHERE source = 'cloudflare_r2'
                """)

            except Exception as e:
                print(f"❌ Error adding detections: {e}")

        return total_detections

    def get_r2_image_stats(self) -> Dict:
        """Get statistics about R2 images in database"""

        stats = {}

        try:
            # Count R2 images
            result = self.conn.execute("""
                SELECT COUNT(*) as count
                FROM images
                WHERE source = 'cloudflare_r2'
            """).fetchone()

            stats['total_images'] = result[0] if result else 0

            # Count images with detections
            result = self.conn.execute("""
                SELECT COUNT(*) as count
                FROM images
                WHERE source = 'cloudflare_r2'
                AND furniture_count > 0
            """).fetchone()

            stats['images_with_detections'] = result[0] if result else 0

            # Count total detections
            result = self.conn.execute("""
                SELECT COUNT(*) as count
                FROM furniture_detections fd
                JOIN images i ON fd.image_id = i.image_id
                WHERE i.source = 'cloudflare_r2'
            """).fetchone()

            stats['total_detections'] = result[0] if result else 0

            # Room type distribution
            room_dist = self.conn.execute("""
                SELECT room_type, COUNT(*) as count
                FROM images
                WHERE source = 'cloudflare_r2'
                GROUP BY room_type
                ORDER BY count DESC
            """).df()

            stats['room_types'] = room_dist.to_dict('records')

            # Style distribution
            style_dist = self.conn.execute("""
                SELECT style, COUNT(*) as count
                FROM images
                WHERE source = 'cloudflare_r2'
                GROUP BY style
                ORDER BY count DESC
            """).df()

            stats['styles'] = style_dist.to_dict('records')

        except Exception as e:
            print(f"❌ Error getting stats: {e}")

        return stats

    def close(self):
        """Close database connection"""
        self.conn.close()


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Download and integrate R2 images into training database'
    )

    parser.add_argument('--db', type=str, required=True,
                        help='Path to DuckDB database')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Directory to download images to')
    parser.add_argument('--prefix', '-p', type=str, default='',
                        help='R2 bucket prefix/folder')
    parser.add_argument('--max-images', type=int, default=None,
                        help='Maximum number of images to download')
    parser.add_argument('--room-type', type=str, default='living_room',
                        help='Default room type for images')
    parser.add_argument('--style', type=str, default='modern',
                        help='Default style for images')
    parser.add_argument('--metadata', type=str, default=None,
                        help='JSON file with image metadata')
    parser.add_argument('--no-detection', action='store_true',
                        help='Skip furniture detection step')
    parser.add_argument('--stats', action='store_true',
                        help='Show R2 image statistics only')

    args = parser.parse_args()

    print("=" * 70)
    print("🌐 CLOUDFLARE R2 IMAGE INTEGRATION")
    print("=" * 70)

    try:
        # Initialize downloader
        downloader = CloudflareR2Downloader()

        # Initialize integrator
        integrator = R2ImageIntegrator(
            db_path=args.db,
            r2_downloader=downloader
        )

        # Show stats if requested
        if args.stats:
            print("\n📊 R2 Image Statistics:")
            stats = integrator.get_r2_image_stats()
            print(f"   Total images: {stats.get('total_images', 0):,}")
            print(f"   Images with detections: {stats.get('images_with_detections', 0):,}")
            print(f"   Total detections: {stats.get('total_detections', 0):,}")

            if stats.get('room_types'):
                print("\n   Room Types:")
                for item in stats['room_types']:
                    print(f"      {item['room_type']}: {item['count']:,}")

            if stats.get('styles'):
                print("\n   Styles:")
                for item in stats['styles']:
                    print(f"      {item['style']}: {item['count']:,}")

            integrator.close()
            exit(0)

        # Download and process images
        result = integrator.download_and_process(
            output_dir=args.output,
            prefix=args.prefix,
            max_images=args.max_images,
            default_room_type=args.room_type,
            default_style=args.style,
            run_detection=not args.no_detection,
            metadata_file=args.metadata
        )

        # Show final stats
        print("\n📊 Final Statistics:")
        final_stats = integrator.get_r2_image_stats()
        print(f"   Total R2 images in DB: {final_stats.get('total_images', 0):,}")
        print(f"   Images with detections: {final_stats.get('images_with_detections', 0):,}")
        print(f"   Total detections: {final_stats.get('total_detections', 0):,}")

        integrator.close()

        print("\n✅ Integration complete! Images are ready for training.")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
