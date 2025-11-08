# ============================================
# ============================================

class FastCLIPClassifier:
    """BATCH processing CLIP classifier - 100x faster!"""
    
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🎮 Using device: {self.device}")
        
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        self.room_prompts = [
            "a photo of a living room",
            "a photo of a bedroom", 
            "a photo of a kitchen",
            "a photo of a bathroom",
            "a photo of a dining room",
            "a photo of a home office"
        ]
        
        self.style_prompts = [
            "modern interior design style",
            "traditional interior design style",
            "minimalist interior design style",
            "scandinavian interior design style",
            "industrial interior design style",
            "bohemian interior design style",
            "contemporary interior design style",
            "rustic interior design style",
            "mid century modern interior design style"
        ]
        
        self._encode_text_prompts()
    
    def _encode_text_prompts(self):
        """Pre-encode text prompts for speed"""
        room_inputs = self.processor(text=self.room_prompts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            self.room_text_features = self.model.get_text_features(**room_inputs)
            self.room_text_features = self.room_text_features / self.room_text_features.norm(dim=-1, keepdim=True)
        
        style_inputs = self.processor(text=self.style_prompts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            self.style_text_features = self.model.get_text_features(**style_inputs)
            self.style_text_features = self.style_text_features / self.style_text_features.norm(dim=-1, keepdim=True)
    
    def classify_batch(self, images: List[Image.Image]) -> List[dict]:
        """Classify multiple images at once - MUCH FASTER!"""
        image_inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**image_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Room classification
            room_logits = torch.matmul(image_features, self.room_text_features.T)
            room_probs = room_logits.softmax(dim=-1).cpu().numpy()
            
            # Style classification
            style_logits = torch.matmul(image_features, self.style_text_features.T)
            style_probs = style_logits.softmax(dim=-1).cpu().numpy()
        
        results = []
        for i in range(len(images)):
            room_idx = np.argmax(room_probs[i])
            room_type = self.room_prompts[room_idx].replace("a photo of a ", "").replace(" ", "_").replace("-", "_")
            
            style_idx = np.argmax(style_probs[i])
            style = self.style_prompts[style_idx].replace(" interior design style", "").replace(" ", "_").replace("-", "_")
            
            results.append({
                "room_type": room_type,
                "room_confidence": float(room_probs[i][room_idx]),
                "style": style,
                "style_confidence": float(style_probs[i][style_idx])
            })
        
        return results

# ============================================
# ============================================

class FastProcessor:
    """Fast batch processor"""
    
    def __init__(self, data_dir: Path, output_dir: Path = None, batch_size: int = 64):
        self.data_dir = Path(data_dir)
        self.output_dir = output_dir or self.data_dir / "processed"
        self.output_dir.mkdir(exist_ok=True)
        
        self.batch_size = batch_size
        self.classifier = FastCLIPClassifier()
        self.metadata_records = []
        
        self.processed_ids = self._load_processed_ids()
    
    def _load_processed_ids(self) -> set:
        """Load already processed image IDs"""
        parquet_path = self.output_dir / "metadata.parquet"
        if parquet_path.exists():
            try:
                df = pd.read_parquet(parquet_path)
                print(f"📝 Found {len(df)} already processed images - will skip these")
                return set(df['image_id'].values)
            except:
                return set()
        return set()
    
    def find_new_images(self) -> List[Path]:
        """Find only NEW images from hybrid collection"""
        print("🔍 Finding NEW images from hybrid collection...")
        
        hybrid_dir = Path("./interior_design_data_hybrid")
        
        if not hybrid_dir.exists():
            print("❌ Hybrid directory not found!")
            return []
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        all_images = []
        
        for ext in image_extensions:
            all_images.extend(hybrid_dir.glob(f"**/*{ext}"))
        
        new_images = []
        for img_path in all_images:
            image_id = hashlib.md5(str(img_path).encode()).hexdigest()[:16]
            if image_id not in self.processed_ids:
                new_images.append(img_path)
        
        print(f"✅ Found {len(new_images)} NEW images to process")
        return new_images
    
    def determine_source(self, image_path: Path) -> tuple:
        """Determine source from path"""
        parts = image_path.parts
        
        source = None
        dataset_name = None
        
        for i, part in enumerate(parts):
            if part in ['huggingface', 'kaggle', 'roboflow', 'unsplash', 'pexels']:
                source = part
                if i + 1 < len(parts):
                    dataset_name = parts[i + 1]
                break
        
        return source or 'unknown', dataset_name or 'unknown'
    
    def process_batch(self, image_paths: List[Path]) -> List[ImageMetadata]:
        """Process a batch of images"""
        images = []
        valid_paths = []
        
        for path in image_paths:
            try:
                img = Image.open(path).convert('RGB')
                images.append(img)
                valid_paths.append(path)
            except:
                continue
        
        if not images:
            return []
        
        classifications = self.classifier.classify_batch(images)
        
        metadata_batch = []
        for i, path in enumerate(valid_paths):
            try:
                source, dataset_name = self.determine_source(path)
                image_id = hashlib.md5(str(path).encode()).hexdigest()[:16]
                
                width, height = images[i].size
                file_size = path.stat().st_size
                
                metadata = ImageMetadata(
                    image_id=image_id,
                    source=source,
                    dataset_name=dataset_name,
                    original_path=str(path),
                    processed_path=str(path),
                    room_type=classifications[i]['room_type'],
                    style=classifications[i]['style'],
                    room_confidence=classifications[i]['room_confidence'],
                    style_confidence=classifications[i]['style_confidence'],
                    dimensions={'width': width, 'height': height},
                    file_size=file_size
                )
                
                metadata_batch.append(metadata)
            except:
                continue
        
        return metadata_batch
    
    def process_all(self):
        """Process all images in batches"""
        image_paths = self.find_new_images()
        
        if not image_paths:
            print("✅ No new images to process!")
            return
        
        print(f"\n🚀 Processing {len(image_paths)} images in batches of {self.batch_size}...")
        
        for i in tqdm(range(0, len(image_paths), self.batch_size), desc="Processing batches"):
            batch_paths = image_paths[i:i+self.batch_size]
            batch_metadata = self.process_batch(batch_paths)
            self.metadata_records.extend(batch_metadata)
            
            if (i // self.batch_size) % 10 == 0 and self.metadata_records:
                self._save_checkpoint()
        
        print(f"\n✅ Successfully processed {len(self.metadata_records)} images")
    
    def _save_checkpoint(self):
        """Save progress checkpoint"""
        if not self.metadata_records:
            return
        
        checkpoint_path = self.output_dir / "metadata_checkpoint.parquet"
        df = pd.DataFrame([m.to_dict() for m in self.metadata_records])
        df.to_parquet(checkpoint_path, index=False)
    
    def save_metadata(self):
        """Save final metadata"""
        print("\n💾 Saving metadata...")
        
        existing_df = None
        parquet_path = self.output_dir / "metadata.parquet"
        if parquet_path.exists():
            existing_df = pd.read_parquet(parquet_path)
        
        new_df = pd.DataFrame([m.to_dict() for m in self.metadata_records])
        
        if existing_df is not None:
            df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            df = new_df
        
        df.to_parquet(parquet_path, index=False)
        print(f"  ✅ Saved Parquet: {parquet_path}")
        
        df.to_csv(self.output_dir / "metadata.csv", index=False)
        print(f"  ✅ Saved CSV")
        
        db_path = self.output_dir / "metadata.duckdb"
        conn = duckdb.connect(str(db_path))
        conn.execute("DROP TABLE IF EXISTS images")
        conn.execute("CREATE TABLE images AS SELECT * FROM df")
        conn.close()
        print(f"  ✅ Saved DuckDB")
        
        return df
    
    def show_stats(self, df: pd.DataFrame):
        """Show statistics"""
        print("\n" + "=" * 70)
        print("📊 DATASET STATISTICS")
        print("=" * 70)
        
        print(f"\n📷 Total: {len(df):,}")
        
        print(f"\n🗂️  By Source:")
        for source, count in df['source'].value_counts().head(10).items():
            print(f"   {source:15s}: {count:5,} ({count/len(df)*100:.1f}%)")
        
        print(f"\n🏠 By Room:")
        for room, count in df['room_type'].value_counts().items():
            print(f"   {room:20s}: {count:5,} ({count/len(df)*100:.1f}%)")
        
        print(f"\n🎨 By Style:")
        for style, count in df['style'].value_counts().items():
            print(f"   {style:20s}: {count:5,} ({count/len(df)*100:.1f}%)")
        
        print("\n" + "=" * 70)

# ============================================
# ============================================

print("🚀 Starting FAST batch processing...\n")

data_dir = Path("./interior_design_data_hybrid")

processor = FastProcessor(data_dir, batch_size=64)

processor.process_all()

df = processor.save_metadata()

processor.show_stats(df)

print("\n🎉 COMPLETE! Ready for training!")