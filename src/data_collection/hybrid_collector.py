"""
Hybrid data collector for gathering interior design images from multiple sources.
Supports HuggingFace, Kaggle, Roboflow, Unsplash, and Pexels.
"""

import os
import json
import time
from pathlib import Path
from typing import List, Optional
from datetime import datetime
from itertools import islice
import warnings
warnings.filterwarnings('ignore')

import requests
from PIL import Image
from tqdm.auto import tqdm
from huggingface_hub import HfApi
from datasets import load_dataset

# Optional imports with fallback
try:
    import kaggle
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False
    kaggle = None

try:
    from roboflow import Roboflow
    ROBOFLOW_AVAILABLE = True
except ImportError:
    ROBOFLOW_AVAILABLE = False
    Roboflow = None

# Import taxonomy
try:
    from interior_taxonomy import (
        INTERIOR_TAXONOMY,
        get_all_categories,
        get_items_by_room,
        get_items_by_style
    )
except ImportError:
    # Fallback if running from different directory
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from interior_taxonomy import (
        INTERIOR_TAXONOMY,
        get_all_categories,
        get_items_by_room,
        get_items_by_style
    )


class HybridConfig:
    """Configuration for hybrid data collection"""
    
    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir) if base_dir else Path("./interior_design_data_hybrid")
        self.base_dir.mkdir(exist_ok=True, parents=True)
        
        self.huggingface_target = 4000
        self.kaggle_target = 4000
        self.roboflow_target = 2000
        self.unsplash_target = 3000
        self.pexels_target = 3000
        
        self.manual_hf_datasets = [
            ("hammer888/interior_style_dataset", None, 500),
            ("keremberke/indoor-scene-classification", "full", 500),
            ("tonijhanel/my_interior_design_dataset", None, 500),
            ("FatimaSaadNaik/indoor-scenes-dataset", None, 500),
            ("pcuenq/lsun-bedrooms", None, 500),
        ]
        
        self.manual_kaggle_datasets = [
            ("robinreni/house-rooms-image-dataset", 2000),
            ("prashantsingh001/bedroom-interior-dataset", 500),
            ("galinakg/interior-design-images-and-metadata", 500),
            ("udaysankarmukherjee/furniture-image-dataset", 300),
            ("itsahmad/indoor-scenes-cvpr-2019", 500),
            ("stepanyarullin/interior-design-styles", 500),
        ]
        
        self.manual_roboflow_projects = [
            "roboflow-100/furniture-ngpea/1",
            "class-qq9at/interiordesign/1",
            "singapore-university-of-technology-and-design/interior-furniture/1",
        ]
        
        self.search_terms = [
            # BASE TERMS
            "interior design", "interior", "room", "furniture", 
            "indoor scene", "home decor", "house interior",
            
            # ROOM TYPES
            "living room", "bedroom", "kitchen", "bathroom", 
            "dining room", "home office", "workspace", "entryway", 
            "hallway", "nursery", "game room", "mudroom", "studio apartment",
            
            # DESIGN STYLES
            "modern interior", "contemporary", "traditional", "minimalist", 
            "scandinavian", "rustic", "industrial", "bohemian", 
            "mid century modern", "farmhouse", "coastal", "luxury",
            "art deco", "french provincial", "transitional", "eclectic",
            "vintage", "shabby chic", "mediterranean", "asian inspired",
            
            # SEATING FURNITURE
            "sectional sofa", "loveseat", "chesterfield", "accent chair",
            "wicker chair", "rattan chair", "wingback chair", "dining chair",
            "bar stool", "ottoman", "pouf", "rocking chair",
            
            # TABLES
            "coffee table", "dining table", "console table", "side table",
            "nightstand", "vanity table", "desk", "end table", "corner table",
            
            # STORAGE
            "bookshelf", "dresser", "credenza", "sideboard", "tv stand",
            "armoire", "wardrobe", "buffet", "media console",
            
            # LIGHTING
            "chandelier", "pendant light", "floor lamp", "table lamp",
            "wall sconce", "ceiling light", "recessed lighting",
            
            # DECORATIVE
            "wall art", "mirror", "decorative mirror", "potted plant",
            "vase", "flower vase", "throw pillow", "area rug",
            "persian rug", "wall hanging", "sculpture",
            
            # TEXTILES
            "curtains", "drapes", "blinds", "roman shades", "shutters",
            "throw blanket", "decorative pillow", "bedding",
            
            # SPECIALTY ITEMS
            "bar cart", "wine rack", "decorative ladder", "trolley",
            "fireplace", "room divider",
            
            # MATERIALS & FINISHES
            "wood furniture", "metal furniture", "glass furniture",
            "marble", "velvet", "leather", "rattan", "wicker"
        ]
        
        self.max_per_hf_dataset = 1000
        self.max_per_kaggle_dataset = 1000
        self.images_per_query = 100
        
        # Store reference to taxonomy
        try:
            self.furniture_categories = get_all_categories()
            self.taxonomy_dict = INTERIOR_TAXONOMY
        except Exception:
            self.furniture_categories = []
            self.taxonomy_dict = {}


class HybridCollector:
    """Collects interior design images from multiple sources"""
    
    def __init__(self, config: HybridConfig):
        self.config = config
        self.stats = {
            'huggingface_manual': 0,
            'huggingface_auto': 0,
            'kaggle_manual': 0,
            'kaggle_auto': 0,
            'roboflow': 0,
            'unsplash': 0,
            'pexels': 0
        }
        
        # Check availability
        self.kaggle_available = KAGGLE_AVAILABLE
        self.roboflow_available = ROBOFLOW_AVAILABLE
        
        # Get API keys from environment
        self.unsplash_key = os.getenv('UNSPLASH_ACCESS_KEY')
        self.pexels_key = os.getenv('PEXELS_API_KEY')
        self.kaggle_username = os.getenv('KAGGLE_USERNAME')
        self.kaggle_key = os.getenv('KAGGLE_KEY')
        self.roboflow_key = os.getenv('ROBOFLOW_API_KEY')
    
    def collect_hf_manual(self) -> int:
        """Collect from manually curated HuggingFace datasets"""
        print("\n" + "=" * 70)
        print("1A️⃣ HUGGINGFACE - MANUAL CURATED LIST")
        print("=" * 70)
        
        output_dir = self.config.base_dir / "huggingface"
        output_dir.mkdir(exist_ok=True, parents=True)
        
        total = 0
        for dataset_name, config_name, max_samples in tqdm(self.config.manual_hf_datasets, desc="HF Manual"):
            try:
                if config_name:
                    dataset = load_dataset(dataset_name, config_name, split="train", streaming=True)
                else:
                    dataset = load_dataset(dataset_name, split="train", streaming=True)
                
                dataset_dir = output_dir / dataset_name.replace("/", "_")
                dataset_dir.mkdir(exist_ok=True, parents=True)
                
                count = 0
                for idx, sample in enumerate(islice(dataset, max_samples)):
                    try:
                        image = None
                        for field in ['image', 'img', 'images', 'photo', 'picture']:
                            if field in sample and sample[field]:
                                image = sample[field]
                                break
                        
                        if image and isinstance(image, Image.Image):
                            filename = f"{dataset_name.replace('/', '_')}_{idx:05d}.jpg"
                            filepath = dataset_dir / filename
                            image.save(filepath, "JPEG", quality=85)
                            count += 1
                            total += 1
                    except Exception:
                        continue
                
                if count > 0:
                    tqdm.write(f"  ✅ {dataset_name}: {count} images")
            except Exception as e:
                tqdm.write(f"  ❌ {dataset_name}: {str(e)[:50]}")
        
        self.stats['huggingface_manual'] = total
        print(f"\n✅ HF Manual: {total} images")
        return total
    
    def collect_hf_auto(self) -> int:
        """Collect from automatically discovered HuggingFace datasets"""
        print("\n" + "=" * 70)
        print("1B️⃣ HUGGINGFACE - AUTOMATED DISCOVERY")
        print("=" * 70)
        
        if self.stats['huggingface_manual'] >= self.config.huggingface_target:
            print(f"✅ Already have {self.stats['huggingface_manual']} - skipping")
            return 0
        
        output_dir = self.config.base_dir / "huggingface"
        output_dir.mkdir(exist_ok=True, parents=True)
        
        api = HfApi()
        manual_ids = set(d[0] for d in self.config.manual_hf_datasets)
        discovered = set()
        
        print("\n🔍 Discovering...")
        for term in self.config.search_terms[:10]:
            try:
                datasets = api.list_datasets(search=term, sort="downloads", direction=-1)
                for dataset in islice(datasets, 5):
                    if dataset.id not in manual_ids:
                        discovered.add(dataset.id)
            except Exception:
                continue
        
        print(f"✅ Found {len(discovered)} datasets\n")
        
        total = 0
        remaining = self.config.huggingface_target - self.stats['huggingface_manual']
        
        for dataset_name in tqdm(list(discovered), desc="HF Auto"):
            if total >= remaining:
                break
            
            try:
                dataset = load_dataset(dataset_name, split="train", streaming=True)
                dataset_dir = output_dir / dataset_name.replace("/", "_")
                dataset_dir.mkdir(exist_ok=True, parents=True)
                
                count = 0
                limit = min(self.config.max_per_hf_dataset, remaining - total)
                
                for idx, sample in enumerate(islice(dataset, limit)):
                    try:
                        image = None
                        for field in ['image', 'img', 'images', 'photo', 'picture']:
                            if field in sample and sample[field]:
                                image = sample[field]
                                break
                        
                        if image and isinstance(image, Image.Image):
                            filename = f"{dataset_name.replace('/', '_')}_{idx:05d}.jpg"
                            filepath = dataset_dir / filename
                            image.save(filepath, "JPEG", quality=85)
                            count += 1
                            total += 1
                    except Exception:
                        continue
                
                if count > 0:
                    tqdm.write(f"  ✅ {dataset_name}: {count} images")
            except Exception:
                continue
        
        self.stats['huggingface_auto'] = total
        print(f"\n✅ HF Auto: {total} images")
        return total
    
    def collect_kaggle_manual(self) -> int:
        """Collect from manually curated Kaggle datasets"""
        print("\n" + "=" * 70)
        print("2A️⃣ KAGGLE - MANUAL CURATED LIST")
        print("=" * 70)
        
        if not self.kaggle_available or not self.kaggle_username or not self.kaggle_key:
            print("⚠️ Kaggle not available")
            return 0
        
        output_dir = self.config.base_dir / "kaggle"
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Setup credentials
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_dir.mkdir(exist_ok=True, parents=True)
        with open(kaggle_dir / "kaggle.json", 'w') as f:
            json.dump({"username": self.kaggle_username, "key": self.kaggle_key}, f)
        os.chmod(kaggle_dir / "kaggle.json", 0o600)
        
        total = 0
        for dataset_ref, max_images in tqdm(self.config.manual_kaggle_datasets, desc="Kaggle Manual"):
            try:
                dataset_dir = output_dir / dataset_ref.replace("/", "_")
                dataset_dir.mkdir(exist_ok=True, parents=True)
                
                kaggle.api.dataset_download_files(dataset_ref, path=str(dataset_dir), unzip=True)
                
                count = 0
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                    for img_path in list(dataset_dir.glob(f"**/{ext}"))[:max_images - count]:
                        count += 1
                        total += 1
                        if count >= max_images:
                            break
                
                if count > 0:
                    tqdm.write(f"  ✅ {dataset_ref}: {count} images")
            except Exception as e:
                tqdm.write(f"  ❌ {dataset_ref}: {str(e)[:50]}")
        
        self.stats['kaggle_manual'] = total
        print(f"\n✅ Kaggle Manual: {total} images")
        return total
    
    def collect_kaggle_auto(self) -> int:
        """Collect from automatically discovered Kaggle datasets"""
        print("\n" + "=" * 70)
        print("2B️⃣ KAGGLE - AUTOMATED DISCOVERY")
        print("=" * 70)
        
        if self.stats['kaggle_manual'] >= self.config.kaggle_target:
            print(f"✅ Already have {self.stats['kaggle_manual']} - skipping")
            return 0
        
        if not self.kaggle_available:
            return 0
        
        output_dir = self.config.base_dir / "kaggle"
        output_dir.mkdir(exist_ok=True, parents=True)
        manual_refs = set(d[0] for d in self.config.manual_kaggle_datasets)
        discovered = []
        
        print("\n🔍 Discovering...")
        for term in self.config.search_terms[:8]:
            try:
                datasets = kaggle.api.dataset_list(search=term, sort_by="hottest")
                for dataset in datasets[:3]:
                    if dataset.ref not in manual_refs:
                        discovered.append((dataset.ref, dataset.title))
            except Exception:
                continue
        
        print(f"✅ Found {len(discovered)} datasets\n")
        
        total = 0
        remaining = self.config.kaggle_target - self.stats['kaggle_manual']
        
        for dataset_ref, title in tqdm(discovered, desc="Kaggle Auto"):
            if total >= remaining:
                break
            
            try:
                dataset_dir = output_dir / dataset_ref.replace("/", "_")
                dataset_dir.mkdir(exist_ok=True, parents=True)
                
                kaggle.api.dataset_download_files(dataset_ref, path=str(dataset_dir), unzip=True)
                
                count = 0
                limit = min(self.config.max_per_kaggle_dataset, remaining - total)
                
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                    for img_path in list(dataset_dir.glob(f"**/{ext}"))[:limit - count]:
                        count += 1
                        total += 1
                        if count >= limit:
                            break
                
                if count > 0:
                    tqdm.write(f"  ✅ {dataset_ref}: {count} images")
            except Exception:
                continue
        
        self.stats['kaggle_auto'] = total
        print(f"\n✅ Kaggle Auto: {total} images")
        return total
    
    def collect_roboflow(self) -> int:
        """Collect from Roboflow projects"""
        print("\n" + "=" * 70)
        print("3️⃣ ROBOFLOW")
        print("=" * 70)
        
        if not self.roboflow_available or not self.roboflow_key:
            print("⚠️ Roboflow not available")
            return 0
        
        output_dir = self.config.base_dir / "roboflow"
        output_dir.mkdir(exist_ok=True, parents=True)
        
        total = 0
        rf = Roboflow(api_key=self.roboflow_key)
        
        for project_name in tqdm(self.config.manual_roboflow_projects, desc="Roboflow"):
            try:
                parts = project_name.split("/")
                workspace, project, version = parts[0], parts[1], int(parts[2])
                
                project_obj = rf.workspace(workspace).project(project)
                project_obj.version(version).download("coco", location=str(output_dir / workspace))
                
                count = len(list((output_dir / workspace).glob("**/*.jpg")))
                total += count
                tqdm.write(f"  ✅ {project_name}: {count} images")
            except Exception:
                continue
        
        self.stats['roboflow'] = total
        print(f"\n✅ Roboflow: {total} images")
        return total
    
    def collect_unsplash(self) -> int:
        """Collect from Unsplash API"""
        print("\n" + "=" * 70)
        print("4️⃣ UNSPLASH")
        print("=" * 70)
        
        if not self.unsplash_key:
            print("⚠️ Unsplash key not found")
            return 0
        
        output_dir = self.config.base_dir / "unsplash"
        output_dir.mkdir(exist_ok=True, parents=True)
        
        total = 0
        queries_needed = (self.config.unsplash_target // self.config.images_per_query) + 1
        
        for query in tqdm(self.config.search_terms[:queries_needed], desc="Unsplash"):
            if total >= self.config.unsplash_target:
                break
            
            try:
                url = "https://api.unsplash.com/search/photos"
                count = 0
                
                for page in range(1, 5):
                    if count >= self.config.images_per_query:
                        break
                    
                    params = {
                        "query": query,
                        "per_page": 30,
                        "page": page,
                        "client_id": self.unsplash_key
                    }
                    
                    response = requests.get(url, params=params, timeout=10)
                    response.raise_for_status()
                    data = response.json()
                    
                    for photo in data['results']:
                        if count >= self.config.images_per_query or total >= self.config.unsplash_target:
                            break
                        
                        try:
                            img_url = photo['urls']['regular']
                            img_id = photo['id']
                            
                            img_response = requests.get(img_url, timeout=10)
                            if img_response.status_code == 200:
                                filename = f"unsplash_{query.replace(' ', '_')}_{img_id}.jpg"
                                filepath = output_dir / filename
                                
                                with open(filepath, 'wb') as f:
                                    f.write(img_response.content)
                                
                                count += 1
                                total += 1
                        except Exception:
                            continue
                    
                    time.sleep(1)
                
                if count > 0:
                    tqdm.write(f"  ✅ {query}: {count} images")
            except Exception:
                continue
        
        self.stats['unsplash'] = total
        print(f"\n✅ Unsplash: {total} images")
        return total
    
    def collect_pexels(self) -> int:
        """Collect from Pexels API"""
        print("\n" + "=" * 70)
        print("5️⃣ PEXELS")
        print("=" * 70)
        
        if not self.pexels_key:
            print("⚠️ Pexels key not found")
            return 0
        
        output_dir = self.config.base_dir / "pexels"
        output_dir.mkdir(exist_ok=True, parents=True)
        
        total = 0
        queries_needed = (self.config.pexels_target // self.config.images_per_query) + 1
        
        for query in tqdm(self.config.search_terms[:queries_needed], desc="Pexels"):
            if total >= self.config.pexels_target:
                break
            
            try:
                url = "https://api.pexels.com/v1/search"
                headers = {"Authorization": self.pexels_key}
                count = 0
                
                for page in range(1, 3):
                    if count >= self.config.images_per_query:
                        break
                    
                    params = {
                        "query": query,
                        "per_page": 80,
                        "page": page
                    }
                    
                    response = requests.get(url, headers=headers, params=params, timeout=10)
                    response.raise_for_status()
                    data = response.json()
                    
                    for photo in data['photos']:
                        if count >= self.config.images_per_query or total >= self.config.pexels_target:
                            break
                        
                        try:
                            img_url = photo['src']['large']
                            img_id = photo['id']
                            
                            img_response = requests.get(img_url, timeout=10)
                            if img_response.status_code == 200:
                                filename = f"pexels_{query.replace(' ', '_')}_{img_id}.jpg"
                                filepath = output_dir / filename
                                
                                with open(filepath, 'wb') as f:
                                    f.write(img_response.content)
                                
                                count += 1
                                total += 1
                        except Exception:
                            continue
                    
                    time.sleep(1)
                
                if count > 0:
                    tqdm.write(f"  ✅ {query}: {count} images")
            except Exception:
                continue
        
        self.stats['pexels'] = total
        print(f"\n✅ Pexels: {total} images")
        return total
    
    def collect_all(self) -> int:
        """Collect from all sources"""
        print("\n" + "=" * 70)
        print("🚀 HYBRID COLLECTION - MANUAL + AUTOMATED")
        print("   Target: 16,000 images")
        print("=" * 70)
        
        start_time = datetime.now()
        
        self.collect_hf_manual()
        self.collect_hf_auto()
        self.collect_kaggle_manual()
        self.collect_kaggle_auto()
        self.collect_roboflow()
        self.collect_unsplash()
        self.collect_pexels()
        
        hf_total = self.stats['huggingface_manual'] + self.stats['huggingface_auto']
        kg_total = self.stats['kaggle_manual'] + self.stats['kaggle_auto']
        total = hf_total + kg_total + self.stats['roboflow'] + self.stats['unsplash'] + self.stats['pexels']
        
        elapsed = (datetime.now() - start_time).total_seconds() / 60
        
        print("\n" + "=" * 70)
        print("🎉 COMPLETE!")
        print("=" * 70)
        print(f"\n📊 RESULTS:")
        print(f"   HuggingFace:  {hf_total:5,} (Manual: {self.stats['huggingface_manual']}, Auto: {self.stats['huggingface_auto']})")
        print(f"   Kaggle:       {kg_total:5,} (Manual: {self.stats['kaggle_manual']}, Auto: {self.stats['kaggle_auto']})")
        print(f"   Roboflow:     {self.stats['roboflow']:5,}")
        print(f"   Unsplash:     {self.stats['unsplash']:5,}")
        print(f"   Pexels:       {self.stats['pexels']:5,}")
        print(f"   {'─' * 50}")
        print(f"   TOTAL:        {total:5,}")
        print(f"\n⏱️  Time: {elapsed:.1f} minutes")
        print(f"📁 Saved to: {self.config.base_dir}")
        
        return total
