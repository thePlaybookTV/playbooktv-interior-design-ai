# Cloudflare R2 Training Integration

Complete guide to using images from Cloudflare R2 bucket for training your interior design AI models.

---

## Overview

This integration allows you to:
1. **Ping/Connect** to your Cloudflare R2 bucket
2. **Download** images from the bucket
3. **Automatically detect** furniture in images using YOLO
4. **Integrate** images into your training database (DuckDB)
5. **Use** R2 images alongside your existing training data

---

## Prerequisites

### 1. Install Dependencies

```bash
pip install boto3 botocore python-dotenv
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

### 2. Get Cloudflare R2 Credentials

1. Log into [Cloudflare Dashboard](https://dash.cloudflare.com)
2. Navigate to **R2 Storage** in the left sidebar
3. Copy your **Account ID** (shown at the top)
4. Create an **R2 API Token**:
   - Click "Manage R2 API Tokens"
   - Click "Create API Token"
   - Name: "Training Data Access"
   - Permissions: "Object Read & Write"
   - Click "Create API Token"
   - **Copy both the Access Key ID and Secret Access Key** (you can't see them again!)
5. Note your **Bucket Name** (the bucket containing your training images)

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
nano .env
```

Add your credentials:
```bash
CLOUDFLARE_ACCOUNT_ID=your_account_id_here
R2_ACCESS_KEY_ID=your_access_key_id_here
R2_SECRET_ACCESS_KEY=your_secret_access_key_here
R2_BUCKET_NAME=your_bucket_name_here
```

**Important**: Never commit your `.env` file to git!

---

## Quick Start

### Step 1: Test Connection

```bash
python -m src.data_collection.cloudflare_r2_downloader --test
```

**Expected output:**
```
✅ Connected to R2 bucket: your-bucket-name
✅ Connection test successful!
```

### Step 2: List Available Images

```bash
python -m src.data_collection.cloudflare_r2_downloader --list
```

**Sample output:**
```
📋 Found 150 images:
   1. living_room_001.jpg
   2. bedroom_002.jpg
   3. kitchen_003.jpg
   ...
```

### Step 3: Download Images (Test with 10 images)

```bash
python -m src.data_collection.cloudflare_r2_downloader \
    --output ./r2_images \
    --max-images 10
```

### Step 4: Integrate into Training Database

```bash
python scripts/integrate_r2_images.py \
    --db ./interior_design_data_hybrid/processed/metadata.duckdb \
    --output ./r2_images \
    --max-images 10 \
    --room-type living_room \
    --style modern
```

**This will:**
- Download images from R2
- Run YOLO detection to find furniture
- Add images and detections to your DuckDB database
- Make them available for training

---

## Detailed Usage

### Download Images Only

```bash
# Download all images
python -m src.data_collection.cloudflare_r2_downloader \
    --output ./r2_images

# Download from specific folder
python -m src.data_collection.cloudflare_r2_downloader \
    --output ./r2_images \
    --prefix "living_rooms/"

# Download limited number
python -m src.data_collection.cloudflare_r2_downloader \
    --output ./r2_images \
    --max-images 100 \
    --workers 20
```

### Full Integration with Metadata

If you have metadata about your images (room types, styles), create a JSON file:

**metadata.json:**
```json
{
  "living_room_001.jpg": {
    "room_type": "living_room",
    "style": "modern",
    "tags": ["sofa", "coffee_table"]
  },
  "bedroom_002.jpg": {
    "room_type": "bedroom",
    "style": "minimalist",
    "tags": ["bed", "nightstand"]
  }
}
```

Then run:
```bash
python scripts/integrate_r2_images.py \
    --db ./interior_design_data_hybrid/processed/metadata.duckdb \
    --output ./r2_images \
    --metadata metadata.json
```

### Advanced Options

```bash
python scripts/integrate_r2_images.py \
    --db ./interior_design_data_hybrid/processed/metadata.duckdb \
    --output ./r2_images \
    --prefix "training_data/" \
    --max-images 500 \
    --room-type living_room \
    --style modern \
    --metadata metadata.json
```

**Options:**
- `--db`: Path to your DuckDB database
- `--output`: Local directory for downloaded images
- `--prefix`: R2 folder/prefix to filter images
- `--max-images`: Limit number of images
- `--room-type`: Default room type (if not in metadata)
- `--style`: Default style (if not in metadata)
- `--metadata`: JSON file with per-image metadata
- `--no-detection`: Skip furniture detection (faster, but no object data)

---

## Training with R2 Images

Once integrated, R2 images are treated like any other training data!

### Step 1: Verify Images are in Database

```bash
python scripts/integrate_r2_images.py \
    --db ./interior_design_data_hybrid/processed/metadata.duckdb \
    --stats
```

**Output:**
```
📊 R2 Image Statistics:
   Total images: 150
   Images with detections: 145
   Total detections: 1,247

   Room Types:
      living_room: 80
      bedroom: 45
      kitchen: 25

   Styles:
      modern: 90
      minimalist: 40
      traditional: 20
```

### Step 2: Run Training

Your existing training scripts will automatically use R2 images!

```bash
# Training script (existing)
python src/models/training.py
```

Or for YOLO dataset preparation:
```bash
python src/models/yolo_dataset_prep.py
```

The training will now include:
- Your original dataset images
- Images downloaded from R2
- All detections (both original and R2)

---

## Workflow Examples

### Example 1: Quick Test with 10 Images

```bash
# 1. Test connection
python -m src.data_collection.cloudflare_r2_downloader --test

# 2. Download 10 images and integrate
python scripts/integrate_r2_images.py \
    --db database_metadata.duckdb \
    --output ./r2_images_test \
    --max-images 10 \
    --room-type living_room \
    --style modern

# 3. Check stats
python scripts/integrate_r2_images.py \
    --db database_metadata.duckdb \
    --stats
```

### Example 2: Production Setup (Full Dataset)

```bash
# 1. Download all images (this may take a while)
python -m src.data_collection.cloudflare_r2_downloader \
    --output ./interior_design_data_hybrid/r2_images \
    --workers 20

# 2. Integrate with metadata
python scripts/integrate_r2_images.py \
    --db ./interior_design_data_hybrid/processed/metadata.duckdb \
    --output ./interior_design_data_hybrid/r2_images \
    --metadata ./r2_metadata.json

# 3. Verify integration
python scripts/integrate_r2_images.py \
    --db ./interior_design_data_hybrid/processed/metadata.duckdb \
    --stats

# 4. Run training
python scripts/run_phase2_training.py \
    --db ./interior_design_data_hybrid/processed/metadata.duckdb \
    --output ./phase2_outputs
```

### Example 3: Incremental Updates

Add new images periodically:

```bash
# Download only new images from specific folder
python scripts/integrate_r2_images.py \
    --db database_metadata.duckdb \
    --output ./r2_images \
    --prefix "new_uploads_2024/" \
    --room-type bedroom \
    --style contemporary
```

The script automatically skips images that already exist in the database!

---

## Paperspace Setup

If you're running on Paperspace, add this to your workflow:

### 1. Add Credentials to Paperspace

In your Paperspace notebook or terminal:

```bash
cd /notebooks/app

# Create .env file
nano .env
```

Add your R2 credentials, then save (Ctrl+O, Enter, Ctrl+X).

### 2. Run Integration

```bash
cd /notebooks/app

# Install dependencies
pip install boto3 botocore python-dotenv

# Test connection
python -m src.data_collection.cloudflare_r2_downloader --test

# Download and integrate
python scripts/integrate_r2_images.py \
    --db database_metadata.duckdb \
    --output ./r2_training_images \
    --max-images 100
```

### 3. Run Training

```bash
# Your existing training command
python scripts/run_phase2_training.py \
    --db database_metadata.duckdb \
    --output ./phase2_outputs
```

---

## Troubleshooting

### Connection Issues

**Error: "Missing required credentials"**

Solution:
```bash
# Check .env file exists
ls -la .env

# Verify contents (don't share output!)
cat .env

# Make sure values don't have quotes
# Wrong: R2_ACCESS_KEY_ID="abc123"
# Right: R2_ACCESS_KEY_ID=abc123
```

**Error: "Access Denied" or "Invalid credentials"**

Solution:
- Verify your R2 API token has "Object Read & Write" permissions
- Check that the bucket name is correct
- Ensure the account ID matches your R2 account

### Download Issues

**Images not downloading**

```bash
# Check if bucket has images
python -m src.data_collection.cloudflare_r2_downloader --list

# Try with verbose output
python scripts/integrate_r2_images.py \
    --db database_metadata.duckdb \
    --output ./r2_images \
    --max-images 1  # Start with just 1 image
```

**Out of disk space**

```bash
# Check available space
df -h

# Download to external storage
python scripts/integrate_r2_images.py \
    --db database_metadata.duckdb \
    --output /mnt/storage/r2_images
```

### Detection Issues

**Error: "Failed to load YOLO model"**

```bash
# Install ultralytics
pip install ultralytics

# Or skip detection (faster)
python scripts/integrate_r2_images.py \
    --db database_metadata.duckdb \
    --output ./r2_images \
    --no-detection
```

### Database Issues

**Images not appearing in training**

```bash
# Check database
python << 'EOF'
import duckdb
conn = duckdb.connect('database_metadata.duckdb')

# Count R2 images
result = conn.execute("""
    SELECT COUNT(*) FROM images WHERE source = 'cloudflare_r2'
""").fetchone()
print(f"R2 Images: {result[0]}")

# Show sample
result = conn.execute("""
    SELECT image_id, room_type, style, furniture_count
    FROM images WHERE source = 'cloudflare_r2' LIMIT 5
""").df()
print(result)
EOF
```

---

## Performance Tips

### 1. Parallel Downloads

Use more workers for faster downloads:
```bash
python -m src.data_collection.cloudflare_r2_downloader \
    --output ./r2_images \
    --workers 20  # Default is 10
```

### 2. Incremental Integration

Download images in batches:
```bash
# Batch 1
python scripts/integrate_r2_images.py \
    --db database_metadata.duckdb \
    --output ./r2_images \
    --max-images 100

# Batch 2 (script auto-skips existing)
python scripts/integrate_r2_images.py \
    --db database_metadata.duckdb \
    --output ./r2_images \
    --max-images 200
```

### 3. Skip Detection for Speed

If you just want images without object detection:
```bash
python scripts/integrate_r2_images.py \
    --db database_metadata.duckdb \
    --output ./r2_images \
    --no-detection  # Much faster!
```

You can run detection later if needed.

---

## API Reference

### CloudflareR2Downloader

```python
from src.data_collection.cloudflare_r2_downloader import CloudflareR2Downloader

# Initialize
downloader = CloudflareR2Downloader(
    account_id="your_account_id",
    access_key_id="your_access_key",
    secret_access_key="your_secret_key",
    bucket_name="your_bucket"
)

# Or use environment variables
downloader = CloudflareR2Downloader()  # Loads from .env

# Test connection
downloader.test_connection()

# List images
images = downloader.list_images(prefix="living_rooms/")

# Download images
downloaded = downloader.download_all_images(
    output_dir="./images",
    max_images=100
)
```

### R2ImageIntegrator

```python
from scripts.integrate_r2_images import R2ImageIntegrator

# Initialize
integrator = R2ImageIntegrator(
    db_path="database.duckdb",
    r2_downloader=downloader
)

# Download and process
stats = integrator.download_and_process(
    output_dir="./images",
    max_images=100,
    default_room_type="living_room",
    run_detection=True
)

# Get statistics
r2_stats = integrator.get_r2_image_stats()
```

---

## Security Best Practices

1. **Never commit .env file**
   - Already in `.gitignore`
   - Use `.env.example` for reference

2. **Use read-only tokens when possible**
   - If you only need to download, create token with "Object Read" only

3. **Rotate credentials periodically**
   - Create new R2 API tokens every few months
   - Delete old tokens after rotation

4. **Limit token scope**
   - Create separate tokens for different environments (dev, prod)
   - Use bucket-specific permissions if available

---

## FAQ

**Q: How much does R2 storage cost?**
A: Cloudflare R2 has no egress fees. Storage is $0.015/GB/month. First 10GB is free.

**Q: Can I use this with AWS S3 instead?**
A: Yes! The code uses boto3 which is S3-compatible. Just change the endpoint URL.

**Q: Will this duplicate my training data?**
A: No, the script tracks which images are from R2 (source='cloudflare_r2') and skips existing images.

**Q: Can I mix R2 images with local images?**
A: Yes! The training automatically uses all images in the database, regardless of source.

**Q: What happens if I delete images from R2?**
A: Local copies remain. If you re-run integration, it will re-download them (unless you delete local copies too).

**Q: Can I organize images in folders in R2?**
A: Yes! Use the `--prefix` parameter to download from specific folders.

---

## Support

If you run into issues:

1. Check this documentation
2. Verify your `.env` configuration
3. Test connection with `--test` flag
4. Try with `--max-images 1` to test a single image
5. Check the [Cloudflare R2 Documentation](https://developers.cloudflare.com/r2/)

---

## Summary

**To integrate R2 images into training:**

```bash
# 1. Setup credentials
cp .env.example .env
nano .env  # Add your credentials

# 2. Test connection
python -m src.data_collection.cloudflare_r2_downloader --test

# 3. Integrate images
python scripts/integrate_r2_images.py \
    --db database_metadata.duckdb \
    --output ./r2_images \
    --room-type living_room \
    --style modern

# 4. Train as usual
python scripts/run_phase2_training.py \
    --db database_metadata.duckdb \
    --output ./phase2_outputs
```

**That's it!** Your R2 images are now part of your training data. 🚀
