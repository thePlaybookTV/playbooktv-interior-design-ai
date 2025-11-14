# Using Pre-Uploaded Models on Paperspace Gradient

This guide shows how to use pre-uploaded YOLO and SAM2 models on Paperspace to skip downloads and start training faster.

## 📦 What to Upload to Paperspace Models

### 1. YOLO Model
- **Model**: `yolov8m.pt` (or your preferred size: yolov8n.pt, yolov8s.pt, yolov8l.pt, yolov8x.pt)
- **Size**: ~50MB (yolov8m)
- **Download locally first**:
  ```bash
  wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt
  ```

### 2. SAM2 Checkpoint
- **Model**: `sam2_hiera_large.pt`
- **Size**: ~900MB
- **Download locally first**:
  ```bash
  mkdir -p checkpoints
  wget -O checkpoints/sam2_hiera_large.pt https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
  ```

### 3. SAM2 Config File (also upload this)
- **File**: `sam2_hiera_l.yaml`
- **Get from**: SAM2 repository or create with config below

## 🚀 How to Use on Paperspace

### Step 1: Upload Models to Paperspace Gradient

1. Go to **Gradient Console** → **Models**
2. Click **Create Model**
3. Upload these files:
   - `yolov8m.pt`
   - `sam2_hiera_large.pt`
   - `sam2_hiera_l.yaml`

4. Note the model paths (Gradient mounts models at `/models/your-model-name/`)

### Step 2: Update Your Processing Script

Use environment variables to point to your uploaded models:

```bash
# In your Paperspace notebook or workspace

# Set model paths (adjust based on your Gradient model names)
export YOLO_MODEL_PATH="/models/yolov8m/yolov8m.pt"
export SAM2_CHECKPOINT="/models/sam2_large/sam2_hiera_large.pt"
export SAM2_CONFIG="/models/sam2_large/sam2_hiera_l.yaml"

# Run processing with custom model paths
python scripts/process_images_in_batches.py \
    --images r2_phase2_outputs/r2_images \
    --db database_r2_full.duckdb \
    --batch-size 50 \
    --yolo-model $YOLO_MODEL_PATH \
    --sam2-checkpoint $SAM2_CHECKPOINT \
    --sam2-config $SAM2_CONFIG
```

### Step 3: Alternative - Symlink Models

If your models are uploaded to Gradient, you can symlink them:

```bash
# Create checkpoints directory
mkdir -p checkpoints

# Symlink SAM2 checkpoint
ln -s /models/sam2_large/sam2_hiera_large.pt checkpoints/sam2_hiera_large.pt
ln -s /models/sam2_large/sam2_hiera_l.yaml checkpoints/sam2_hiera_l.yaml

# Symlink YOLO model
ln -s /models/yolov8m/yolov8m.pt ./yolov8m.pt

# Now run normally - models are "found" at expected locations
python scripts/process_images_in_batches.py \
    --images r2_phase2_outputs/r2_images \
    --db database_r2_full.duckdb
```

## 📝 SAM2 Config File Content

If you need to create `sam2_hiera_l.yaml`, here's the content:

```yaml
# SAM 2 Hiera Large configuration

image_encoder:
  scalp: 1
  trunk:
    _target_: src.models.sam2.modeling.backbones.hieradet.Hiera
    embed_dim: 144
    num_heads: 2
    drop_path_rate: 0.1
    stages: [2, 6, 36, 4]
    global_att_blocks: [23, 33, 43]
    window_pos_embed_bkg_spatial_size: [7, 7]
  neck:
    _target_: src.models.sam2.modeling.backbones.image_encoder.FpnNeck
    position_encoding:
      _target_: src.models.sam2.modeling.position_encoding.PositionEmbeddingSine
      num_pos_feats: 256
      normalize: true
      scale: null
      temperature: 10000
    d_model: 256
    backbone_channel_list: [1152, 576, 288, 144]
    fpn_top_down_levels: [2, 3]
    fpn_interp_model: nearest

memory_attention:
  _target_: src.models.sam2.modeling.memory_attention.MemoryAttention
  d_model: 256
  pos_enc_at_input: true
  layer:
    _target_: src.models.sam2.modeling.memory_attention.MemoryAttentionLayer
    activation: relu
    dim_feedforward: 2048
    dropout: 0.1
    pos_enc_at_attn: false
    self_attention:
      _target_: src.models.sam2.modeling.sam.transformer.RoPEAttention
      rope_theta: 10000.0
      feat_sizes: [32, 32]
      embedding_dim: 256
      num_heads: 1
      downsample_rate: 1
      dropout: 0.1
    d_model: 256
    pos_enc_at_cross_attn_keys: true
    pos_enc_at_cross_attn_queries: false
    cross_attention:
      _target_: src.models.sam2.modeling.sam.transformer.RoPEAttention
      rope_theta: 10000.0
      feat_sizes: [32, 32]
      rope_k_repeat: True
      embedding_dim: 256
      num_heads: 1
      downsample_rate: 1
      dropout: 0.1
      kv_in_dim: 64
  num_layers: 4

memory_encoder:
  _target_: src.models.sam2.modeling.memory_encoder.MemoryEncoder
  out_dim: 64
  position_encoding:
    _target_: src.models.sam2.modeling.position_encoding.PositionEmbeddingSine
    num_pos_feats: 64
    normalize: true
    scale: null
    temperature: 10000
  mask_downsampler:
    _target_: src.models.sam2.modeling.memory_encoder.MaskDownSampler
    kernel_size: 3
    stride: 2
    padding: 1
  fuser:
    _target_: src.models.sam2.modeling.memory_encoder.Fuser
    layer:
      _target_: src.models.sam2.modeling.memory_encoder.CXBlock
      dim: 256
      kernel_size: 7
      padding: 3
      layer_scale_init_value: 1e-6
      use_dwconv: True
    num_layers: 2

num_maskmem: 7
image_size: 1024
backbone_stride: 16
sam_mask_decoder_extra_args:
  dynamic_multimask_via_stability: true
  dynamic_multimask_stability_delta: 0.05
  dynamic_multimask_stability_thresh: 0.98
compile_image_encoder: False
```

## ⚡ Performance Benefits

**Before (with downloads):**
- YOLO download: ~1 minute
- SAM2 download: ~3-5 minutes
- **Total startup delay: 4-6 minutes**

**After (pre-uploaded models):**
- Model loading from Gradient storage: ~10-30 seconds
- **Total startup delay: < 1 minute**

**Savings: 5+ minutes per run** ⏱️

## 🔧 Troubleshooting

### "Model not found" error
- Check your Gradient model name matches the path
- Use `ls /models/` to see available models

### "Config file not found" error
- Make sure you uploaded `sam2_hiera_l.yaml` with the checkpoint
- Or use the config content above to create it

### "Permission denied" error
- Use symlinks instead of copying: `ln -s /models/... ./...`
- Or set environment variables and update code to read from them

## 📊 Example: Full R2 Processing Command

```bash
#!/bin/bash
# run_r2_processing.sh

# Link uploaded models
mkdir -p checkpoints
ln -sf /models/sam2_large/sam2_hiera_large.pt checkpoints/sam2_hiera_large.pt
ln -sf /models/sam2_large/sam2_hiera_l.yaml checkpoints/sam2_hiera_l.yaml
ln -sf /models/yolov8m/yolov8m.pt ./yolov8m.pt

# Process R2 images
python scripts/process_images_in_batches.py \
    --images r2_phase2_outputs/r2_images \
    --db database_r2_full.duckdb \
    --batch-size 50 \
    --output ./processed

echo "✅ Processing complete!"
```

Make it executable and run:
```bash
chmod +x run_r2_processing.sh
./run_r2_processing.sh
```
