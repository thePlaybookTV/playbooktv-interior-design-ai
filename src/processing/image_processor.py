# ============================================
# ============================================

import os
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List
from datetime import datetime
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from sklearn.cluster import KMeans

# Import taxonomy with fallback
try:
    from interior_taxonomy import (
        get_all_categories,
        get_items_by_room,
        get_items_by_style,
        get_category_type
    )
except ImportError:
    # Fallback if running from different directory
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from interior_taxonomy import (
        get_all_categories,
        get_items_by_room,
        get_items_by_style,
        get_category_type
    )

# ============================================
# ============================================

class DataConfig:
    """Configuration for data processing"""
    def __init__(self):
        self.base_dir = Path("./interior_design_data_hybrid")
        self.processed_images_dir = self.base_dir / "processed_images"
        self.processed_images_dir.mkdir(exist_ok=True)
        
        self.embeddings_dir = self.base_dir / "embeddings"
        self.embeddings_dir.mkdir(exist_ok=True)
        
        self.target_size = (512, 512)  # Standard size for all images
        self.quality = 85  # JPEG quality
        
        self.room_types = [
            "living_room", "bedroom", "kitchen", "dining_room",
            "bathroom", "home_office", "entryway", "hallway", "unknown"
        ]
        
        self.design_styles = [
            "modern", "contemporary", "traditional", "minimalist",
            "scandinavian", "rustic", "industrial", "bohemian",
            "mid_century", "farmhouse", "coastal", "luxury", "unknown"
        ]

# ============================================
# ============================================

@dataclass
class ImageMetadata:
    """Metadata for each image"""
    image_id: str
    source: str
    dataset_name: str
    original_path: str
    processed_path: Optional[str] = None
    room_type: Optional[str] = None
    style: Optional[str] = None
    room_confidence: Optional[float] = None
    style_confidence: Optional[float] = None
    dimensions: Optional[Dict[str, int]] = None
    color_palette: Optional[List[str]] = None
    embedding_path: Optional[str] = None
    file_size: Optional[int] = None
    timestamp: str = datetime.now().isoformat()
    
    def to_dict(self):
        return asdict(self)

# ============================================
# ============================================

class CLIPClassifier:
    """Simple CLIP-based classifier for room type and style"""
    
    def __init__(self):
        from transformers import CLIPProcessor, CLIPModel
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = "openai/clip-vit-base-patch32"
        
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        self.room_templates = [
            "a photo of a {} interior",
            "an interior design photo of a {}",
            "a {} room"
        ]
        
        self.style_templates = [
            "a {} style interior",
            "{} interior design",
            "a {} home decor"
        ]
    
    def classify(self, image: Image.Image, labels: List[str], templates: List[str]) -> tuple:
        """Classify image using CLIP"""
        text_inputs = []
        for label in labels:
            for template in templates:
                text_inputs.append(template.format(label.replace('_', ' ')))
        
        inputs = self.processor(
            text=text_inputs,
            images=image,
            return_tensors="pt",
            padding=True
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
        
        probs_per_label = []
        for i in range(len(labels)):
            start_idx = i * len(templates)
            end_idx = start_idx + len(templates)
            avg_prob = probs[start_idx:end_idx].mean()
            probs_per_label.append(avg_prob)
        
        best_idx = np.argmax(probs_per_label)
        return labels[best_idx], float(probs_per_label[best_idx])
    
    def classify_room(self, image: Image.Image, room_types: List[str]) -> tuple:
        """Classify room type"""
        return self.classify(image, room_types, self.room_templates)
    
    def classify_style(self, image: Image.Image, styles: List[str]) -> tuple:
        """Classify design style"""
        return self.classify(image, styles, self.style_templates)

# ============================================
# ============================================

class ImageProcessor:
    """Handles image preprocessing and feature extraction"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.transform = transforms.Compose([
            transforms.Resize(config.target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print("🤖 Initializing CLIP classifier...")
        self.clip_classifier = CLIPClassifier()
        print("✅ CLIP classifier ready!")
    
    def process_image(self, image_path: str, metadata: ImageMetadata) -> ImageMetadata:
        """Process a single image"""
        try:
            image = Image.open(image_path).convert('RGB')
            
            metadata.dimensions = {
                "original_width": image.width,
                "original_height": image.height
            }
            
            metadata.file_size = os.path.getsize(image_path)
            
            processed_img = image.resize(self.config.target_size, Image.Resampling.LANCZOS)
            processed_path = self.config.processed_images_dir / f"processed_{metadata.image_id}.jpg"
            processed_img.save(processed_path, "JPEG", quality=self.config.quality)
            metadata.processed_path = str(processed_path)
            
            metadata.color_palette = self.extract_color_palette(processed_img)
            
            if not metadata.room_type:
                room_type, confidence = self.detect_room_type(processed_img)
                metadata.room_type = room_type
                metadata.room_confidence = confidence
            
            if not metadata.style:
                style, confidence = self.detect_style(processed_img)
                metadata.style = style
                metadata.style_confidence = confidence
            
            return metadata
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return metadata
    
    def extract_color_palette(self, image: Image.Image, n_colors: int = 5) -> List[str]:
        """Extract dominant colors from image"""
        small_img = image.resize((150, 150))
        pixels = np.array(small_img).reshape(-1, 3)
        
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        colors = []
        for color in kmeans.cluster_centers_:
            hex_color = '#{:02x}{:02x}{:02x}'.format(
                int(color[0]), int(color[1]), int(color[2])
            )
            colors.append(hex_color)
        
        return colors
    
    def detect_room_type(self, image: Image.Image) -> tuple:
        """Detect room type using CLIP"""
        return self.clip_classifier.classify_room(image, self.config.room_types)
    
    def detect_style(self, image: Image.Image) -> tuple:
        """Detect design style using CLIP"""
        return self.clip_classifier.classify_style(image, self.config.design_styles)

# ============================================
# ============================================

def enrich_image_metadata_with_taxonomy(image_metadata: dict) -> dict:
    """
    Add taxonomy-based metadata to an image
    
    Args:
        image_metadata: Dictionary containing image info
    
    Returns:
        Enhanced metadata with taxonomy suggestions
    """
    room_type = image_metadata.get('room_type', 'living_room')
    style = image_metadata.get('style', 'modern')
    
    expected_items = get_items_by_room(room_type)
    style_matched_items = get_items_by_style(style)
    
    image_metadata['taxonomy_expected_items'] = expected_items[:20]  # Top 20
    image_metadata['taxonomy_style_items'] = style_matched_items[:20]
    
    return image_metadata
