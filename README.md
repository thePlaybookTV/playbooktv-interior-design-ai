# 🏠 PlaybookTV Interior Design AI Pipeline

**Advanced AI system for interior design analysis, object detection, and room/style classification.**

## 📋 Project Overview

This project implements a complete end-to-end pipeline for analyzing interior design images, featuring:

- **Multi-Source Data Collection**: Automated collection from HuggingFace, Kaggle, Roboflow, Unsplash, and Pexels
- **Comprehensive Taxonomy**: 294 specific furniture and decor categories across 9 major categories
- **Advanced Object Detection**: YOLO + SAM2 for precise object detection and segmentation
- **Room & Style Classification**: Deep learning models for room type (6 classes) and design style (9 classes)
- **Metadata Management**: DuckDB-based system for efficient metadata storage and querying

## 🎯 Key Features

### 1. Data Collection System
- ✅ **15,000+ images** collected from multiple sources
- ✅ Automated discovery and download pipelines
- ✅ Deduplication and quality filtering
- ✅ Multi-source metadata enrichment

### 2. Interior Taxonomy
- ✅ **294 furniture categories** organized hierarchically
- ✅ 9 main categories: Seating, Tables, Storage, Lighting, Decorative, Textiles, Electronics, Specialty, Architectural
- ✅ Room-type associations (living_room, bedroom, kitchen, etc.)
- ✅ Style-tag mappings (modern, traditional, bohemian, etc.)

### 3. Object Detection & Segmentation
- ✅ **YOLOv8** for fast bounding box detection
- ✅ **SAM2** for precise pixel-level segmentation
- ✅ **100% mask coverage** on all detections
- ✅ Spatial feature extraction for enhanced analysis

### 4. Classification Models
- ✅ **Room Classification**: 68.7% validation accuracy (6 classes)
- ✅ **Style Classification**: 53.8% validation accuracy (9 classes)
- ✅ ResNet18-based architecture with multi-task learning
- ✅ Early stopping and gradient clipping for stability

## 📊 Performance Metrics

| Metric | Training | Validation |
|--------|----------|------------|
| **Room Classification** | 70.1% | 68.7% |
| **Style Classification** | 52.0% | 53.8% |
| **Object Detection** | 100% SAM2 masks | 25,497 detections |
| **Dataset Size** | 4,209 images | 1,053 images |

## 🏗️ Project Structure
```
playbooktv-interior-design-ai/
│
├── src/
│   ├── data_collection/      # Data collection scripts
│   ├── processing/            # Image processing pipeline
│   ├── models/                # ML model definitions
│   └── utils/                 # Utility functions
│
├── notebooks/                 # Jupyter notebooks
│   └── playbooktv_audit.ipynb
│
├── data/                      # Data directory (gitignored)
│   └── interior_design_data_hybrid/
│
├── models/                    # Trained models (gitignored)
│   └── best_interior_model.pth
│
├── docs/                      # Documentation
│   ├── API.md
│   ├── DEPLOYMENT.md
│   └── PRODUCTION_HANDOVER.md
│
├── config/                    # Configuration files
│   └── config.yaml
│
├── tests/                     # Unit tests
│
├── interior_taxonomy.py       # Furniture taxonomy module
├── requirements.txt           # Python dependencies
├── .gitignore                # Git ignore rules
└── README.md                 # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM
- 50GB+ storage

### Installation
```bash
# Clone the repository
git clone https://github.com/thePlaybookTV/playbooktv-interior-design-ai.git
cd playbooktv-interior-design-ai

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Required API Keys

Create a `.env` file with:
```env
# Data Source APIs
HUGGINGFACE_TOKEN=your_token_here
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_key
ROBOFLOW_API_KEY=your_key
UNSPLASH_ACCESS_KEY=your_key
PEXELS_API_KEY=your_key

# AWS (Optional)
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
```

## 📖 Usage

### 1. Data Collection
```python
from src.data_collection.hybrid_collector import HybridCollector, HybridConfig

# Configure collection
config = HybridConfig()
config.huggingface_target = 4000
config.kaggle_target = 4000

# Run collection
collector = HybridCollector(config)
total_images = collector.collect_all()
```

### 2. Image Processing
```python
from src.processing.batch_processor import BatchProcessor

# Process images
processor = BatchProcessor(db_path="./data/metadata.duckdb")
processor.process_all_in_batches(batch_size=64)
```

### 3. Object Detection
```python
from src.models.pristine_detector import PristineDetector

# Initialize detector
detector = PristineDetector()

# Detect objects
results = detector.detect_with_masks("path/to/image.jpg")
```

### 4. Training Classification Models
```python
from src.models.training import train_model

# Train room & style classifier
history, model = train_model(
    db_path="./data/metadata.duckdb",
    num_epochs=15,
    batch_size=16
)
```

## 🗄️ Database Schema

The project uses DuckDB with the following schema:

### `images` Table
```sql
CREATE TABLE images (
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
);
```

### `furniture_detections` Table
```sql
CREATE TABLE furniture_detections (
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
);
```

## 🎨 Interior Taxonomy

The taxonomy includes **294 specific furniture types** organized as:

### Categories (9)
1. **Seating** (40+ types): sectional_sofa, wingback_chair, ottoman, etc.
2. **Tables** (30+ types): coffee_table, console_table, nightstand, etc.
3. **Storage** (35+ types): bookshelf, credenza, armoire, etc.
4. **Lighting** (40+ types): chandelier, pendant_light, floor_lamp, etc.
5. **Decorative** (50+ types): wall_art, vase, potted_plant, etc.
6. **Textiles** (30+ types): curtains, area_rug, throw_pillow, etc.
7. **Electronics** (25+ types): tv, laptop, speakers, etc.
8. **Specialty** (20+ types): bar_cart, fireplace, room_divider, etc.
9. **Architectural** (15+ types): hardwood_floor, bay_window, etc.

### Room Types (7)
- living_room
- bedroom
- kitchen
- dining_room
- bathroom
- home_office
- entryway

### Design Styles (9)
- modern
- traditional
- contemporary
- minimalist
- scandinavian
- industrial
- bohemian
- mid_century_modern
- rustic

## 🧪 Testing
```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_data_collection.py
```

## 📈 Performance Optimization

### Data Collection
- Parallel downloads from multiple sources
- Intelligent rate limiting
- Automatic retry with exponential backoff

### Image Processing
- GPU-accelerated batch processing
- Efficient memory management
- Checkpointing for resumable processing

### Model Inference
- Mixed precision training
- Gradient clipping for stability
- Early stopping to prevent overfitting

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is proprietary to PlaybookTV. All rights reserved.

## 👥 Team

- **Data Science Lead**: [Your Name]
- **Organization**: PlaybookTV
- **Contact**: [Your Email]

## 🙏 Acknowledgments

- **YOLOv8** by Ultralytics
- **SAM2** by Meta AI Research
- **CLIP** by OpenAI
- **HuggingFace** for dataset infrastructure
- **DuckDB** for efficient data management

## 📊 Roadmap

### ✅ Completed (Phase 1)
- [x] Multi-source data collection
- [x] Comprehensive taxonomy creation
- [x] YOLO + SAM2 integration
- [x] Room & style classification
- [x] DuckDB metadata system

### 🔄 In Progress (Phase 2)
- [ ] Fine-tune YOLO on custom taxonomy
- [ ] Web API deployment
- [ ] Real-time inference pipeline
- [ ] User feedback system

### 📅 Planned (Phase 3)
- [ ] Mobile app integration
- [ ] 3D room reconstruction
- [ ] Style recommendation engine
- [ ] Augmented reality features

## 🐛 Known Issues

1. **YOLO detections are generic**: Currently uses COCO classes (chair, bed) instead of specific taxonomy (wingback_chair, platform_bed). Resolution planned in Phase 2.

2. **Style classification accuracy**: 53.8% accuracy is decent but can be improved with more training data.

## 📧 Support

For questions or issues:
- Email: [your-email@playbooktv.com]
- GitHub Issues: [Create an issue](https://github.com/thePlaybookTV/playbooktv-interior-design-ai/issues)

---

**Built with ❤️ by the PlaybookTV Data Science Team**
