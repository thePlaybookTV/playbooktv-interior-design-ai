"""
Batch processing module for processing images in bulk with DuckDB storage.
"""

import os
import json
import hashlib
from pathlib import Path
from typing import List, Optional
from datetime import datetime
from tqdm.auto import tqdm
import duckdb

from .image_processor import ImageProcessor, DataConfig, ImageMetadata


class BatchProcessor:
    """Processes images in batches and stores metadata in DuckDB"""
    
    def __init__(self, db_path: str = "./data/metadata.duckdb", config: Optional[DataConfig] = None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True, parents=True)
        self.config = config or DataConfig()
        self.processor = ImageProcessor(self.config)
        self.conn = duckdb.connect(str(self.db_path))
        self._create_tables()
    
    def _create_tables(self):
        """Create database tables if they don't exist"""
        # Images table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS images (
                image_id VARCHAR PRIMARY KEY,
                source VARCHAR,
                dataset_name VARCHAR,
                original_path VARCHAR,
                processed_path VARCHAR,
                room_type VARCHAR,
                style VARCHAR,
                room_confidence FLOAT,
                style_confidence FLOAT,
                furniture_count INTEGER,
                dimensions JSON,
                color_palette JSON,
                timestamp TIMESTAMP
            )
        """)
        
        # Furniture detections table
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
        
        self.conn.commit()
    
    def _generate_image_id(self, image_path: str) -> str:
        """Generate unique ID for image"""
        with open(image_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return file_hash[:16]
    
    def process_image_file(self, image_path: str, source: str = "unknown", dataset_name: str = "unknown") -> Optional[ImageMetadata]:
        """Process a single image file"""
        try:
            image_id = self._generate_image_id(image_path)
            
            # Check if already processed
            result = self.conn.execute(
                "SELECT image_id FROM images WHERE image_id = ?",
                [image_id]
            ).fetchone()
            
            if result:
                return None  # Already processed
            
            # Create metadata
            metadata = ImageMetadata(
                image_id=image_id,
                source=source,
                dataset_name=dataset_name,
                original_path=str(image_path)
            )
            
            # Process image
            metadata = self.processor.process_image(str(image_path), metadata)
            
            # Store in database
            self.conn.execute("""
                INSERT INTO images (
                    image_id, source, dataset_name, original_path, processed_path,
                    room_type, style, room_confidence, style_confidence,
                    dimensions, color_palette, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                metadata.image_id,
                metadata.source,
                metadata.dataset_name,
                metadata.original_path,
                metadata.processed_path,
                metadata.room_type,
                metadata.style,
                metadata.room_confidence,
                metadata.style_confidence,
                json.dumps(metadata.dimensions) if metadata.dimensions else None,
                json.dumps(metadata.color_palette) if metadata.color_palette else None,
                metadata.timestamp
            ])
            
            self.conn.commit()
            return metadata
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def process_directory(self, directory: Path, source: str = "local", dataset_name: str = "unknown", 
                         extensions: List[str] = None) -> int:
        """Process all images in a directory"""
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        
        image_files = []
        for ext in extensions:
            image_files.extend(directory.rglob(f"*{ext}"))
        
        total = 0
        for image_path in tqdm(image_files, desc=f"Processing {directory.name}"):
            metadata = self.process_image_file(image_path, source, dataset_name)
            if metadata:
                total += 1
        
        return total
    
    def process_all_in_batches(self, base_dir: Optional[Path] = None, batch_size: int = 64) -> int:
        """Process all images from the hybrid collection directory"""
        if base_dir is None:
            base_dir = self.config.base_dir
        
        base_dir = Path(base_dir)
        if not base_dir.exists():
            print(f"Directory {base_dir} does not exist")
            return 0
        
        total = 0
        sources = ['huggingface', 'kaggle', 'roboflow', 'unsplash', 'pexels']
        
        for source in sources:
            source_dir = base_dir / source
            if source_dir.exists():
                count = self.process_directory(source_dir, source=source, dataset_name=source)
                total += count
                print(f"✅ {source}: {count} images processed")
        
        return total
    
    def close(self):
        """Close database connection"""
        self.conn.close()

