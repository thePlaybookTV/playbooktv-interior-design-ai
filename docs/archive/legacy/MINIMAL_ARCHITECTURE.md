# Minimal Architecture: Railway + Modal

**Decision**: Keep Railway backend minimal, move ALL heavy processing to Modal

---

## 🏗️ Architecture Split

### Railway API (Minimal ~500MB)

**Responsibilities**:
- ✅ Receive image uploads
- ✅ Manage job queue (Redis)
- ✅ WebSocket progress updates
- ✅ Communication with Modal
- ✅ Storage management (R2)
- ✅ Basic image operations (resize, optimize)

**Does NOT include**:
- ❌ YOLO
- ❌ SAM2
- ❌ Phase1 classifier
- ❌ PyTorch (except minimal if needed)
- ❌ Transformers
- ❌ Heavy ML models

**Dependencies**: `requirements-railway-minimal.txt`

---

### Modal (GPU Processing)

**Responsibilities**:
- ✅ YOLO + SAM2 detection
- ✅ Phase1 room classification
- ✅ Depth map generation (DPT-Large)
- ✅ Canny edge detection
- ✅ Segmentation processing
- ✅ SD 1.5 + ControlNet transformation
- ✅ Quality validation
- ✅ Result upload to R2

**Includes**:
- ✅ All heavy ML models
- ✅ GPU compute (T4)
- ✅ Auto-scaling (0-10 containers)

**Dependencies**: Full requirements in Modal container

---

## 📊 Processing Flow

```
Mobile App (React Native)
    ↓ Upload image + style selection
Railway API (Minimal)
    ↓
┌───────────────────────────────────────────┐
│ 1. Receive upload                         │
│ 2. Optimize image                         │
│ 3. Upload to R2                           │
│ 4. Create job in Redis                    │
│ 5. Submit to Modal                        │
│ 6. Return job_id                          │
└───────────────────────────────────────────┘
    ↓ job_id
Mobile App
    ↓ WebSocket connection
Railway API (WebSocket)
    ↓
┌───────────────────────────────────────────┐
│ Subscribe to Redis PubSub                 │
│ Stream updates to mobile app              │
└───────────────────────────────────────────┘
    ↑ Updates from Modal
Modal GPU (Complete Processing)
    ↓
┌───────────────────────────────────────────┐
│ 1. Download image from R2                 │
│ 2. Run YOLO + SAM2 detection             │
│ 3. Run Phase1 room classification        │
│ 4. Generate depth map (DPT)              │
│ 5. Generate Canny edges                  │
│ 6. Process segmentation maps             │
│ 7. Run SD 1.5 + ControlNet               │
│ 8. Quality validation                     │
│ 9. Upload result to R2                    │
│ 10. Update Redis with result URL          │
└───────────────────────────────────────────┘
    ↓ result_url
Mobile App
    ↓ Display transformed room
```

---

## 📁 File Structure

### Railway Files (Deployed)

```
api/
  main.py               # Minimal API (new endpoints)
src/
  services/
    job_queue.py        # Redis job management ✅
    storage_service.py  # R2 storage ✅
    control_generators_minimal.py  # Basic image ops ✅
    modal_service.py    # Modal communication (to build)
    websocket_manager.py  # WebSocket handler (to build)
requirements-railway-minimal.txt  # Lightweight deps ✅
Dockerfile              # Updated for minimal ✅
Procfile                # Railway start command ✅
railway.json            # Railway config ✅
```

### Modal Files (Deployed Separately)

```
modal_functions/
  sd_inference_complete.py  # Complete processing pipeline (to build)
    - YOLO + SAM2
    - Phase1 classifier
    - Depth generation
    - Edge detection
    - SD 1.5 + ControlNet
    - Result upload
requirements-modal.txt  # Full ML dependencies (to create)
```

---

## 🆕 New API Endpoints

### 1. POST /transform/submit

**Purpose**: Submit new transformation job

**Request**:
```json
{
  "file": <multipart/form-data>,
  "style": "modern" | "scandinavian" | "boho" | "industrial" | "minimalist",
  "preferences": {
    "keep_furniture": false,
    "color_palette": ["#FFFFFF", "#000000"]  // optional
  }
}
```

**Response**:
```json
{
  "success": true,
  "job_id": "uuid-here",
  "estimated_time": 15,  // seconds
  "websocket_url": "wss://api.../ws/transform/uuid-here",
  "status_url": "https://api.../transform/status/uuid-here"
}
```

**Processing**:
1. Validate image (format, size)
2. Optimize and upload to R2
3. Create Redis job
4. Submit to Modal
5. Return job_id

---

### 2. GET /transform/status/{job_id}

**Purpose**: Check transformation status (polling)

**Response**:
```json
{
  "success": true,
  "data": {
    "job_id": "uuid-here",
    "status": "queued" | "analyzing" | "generating" | "completed" | "failed",
    "progress": 0.75,  // 0.0 - 1.0
    "estimated_time_remaining": 5,  // seconds
    "result": {  // when completed
      "transformed_url": "https://cdn.../result.jpg",
      "thumbnail_url": "https://cdn.../thumb.jpg",
      "original_url": "https://cdn.../original.jpg",
      "metadata": {
        "style": "modern",
        "processing_time": 14.5,
        "quality_score": 0.92
      }
    },
    "error": null  // or error message if failed
  }
}
```

---

### 3. WebSocket /ws/transform/{job_id}

**Purpose**: Real-time progress updates

**Messages from Server**:
```json
{
  "job_id": "uuid-here",
  "status": "analyzing",
  "progress": 0.2,
  "message": "Detecting furniture in your room...",
  "timestamp": "2025-11-12T10:30:45Z"
}
```

**Progress Messages**:
1. `queued` (0.0): "Your transformation is queued..."
2. `analyzing` (0.1-0.2): "Analyzing your room..."
3. `detecting` (0.2-0.3): "Detecting furniture..."
4. `generating` (0.3-0.5): "Generating control images..."
5. `transforming` (0.5-0.9): "Applying modern style..."
6. `finalizing` (0.9-0.95): "Finalizing your design..."
7. `completed` (1.0): "Your new room is ready!"

---

### 4. DELETE /transform/{job_id}

**Purpose**: Cancel pending transformation

**Response**:
```json
{
  "success": true,
  "message": "Job cancelled successfully"
}
```

---

## 🔧 Services to Build

### 1. Modal Service (Railway)

**File**: `src/services/modal_service.py`

```python
class ModalService:
    """
    Interface to Modal GPU processing
    """
    def __init__(self):
        self.modal_client = modal.Client()
        self.stub = modal.Stub.lookup("modomo-sd-inference")

    async def submit_transformation(
        self,
        job_id: str,
        image_url: str,  # R2 URL
        style: str,
        preferences: dict
    ):
        """
        Submit job to Modal for processing
        """
        # Call Modal function
        # Modal processes everything and updates Redis
        pass
```

---

### 2. WebSocket Manager (Railway)

**File**: `src/services/websocket_manager.py`

```python
class WebSocketManager:
    """
    Manages WebSocket connections for real-time updates
    """
    def __init__(self):
        self.active_connections = {}
        self.redis_subscriber = redis.Redis().pubsub()

    async def connect(self, job_id: str, websocket: WebSocket):
        """
        Accept WebSocket connection and subscribe to job updates
        """
        # Subscribe to Redis pub/sub
        # Forward updates to WebSocket
        pass
```

---

### 3. Modal Complete Processing (Modal)

**File**: `modal_functions/sd_inference_complete.py`

```python
@stub.function(gpu="T4", timeout=120)
async def process_transformation_complete(
    job_id: str,
    image_url: str,
    style: str,
    preferences: dict,
    redis_url: str
):
    """
    Complete transformation pipeline on Modal GPU

    Steps:
    1. Download image from R2
    2. YOLO + SAM2 detection
    3. Phase1 room classification
    4. Generate control images (depth, edges)
    5. SD 1.5 + ControlNet transformation
    6. Quality validation
    7. Upload to R2
    8. Update Redis with result
    """
    # Implementation...
    pass
```

---

## 💰 Cost Breakdown

### Railway (Minimal)

- **Plan**: Hobby ($5/month)
- **Resources**: 512MB RAM, 1GB disk
- **Includes**: 500 hours/month
- **Perfect for**: API + WebSocket + Redis client

### Modal (GPU)

- **GPU**: T4 ($0.30/hour)
- **Processing**: ~12 seconds/image
- **Cost per image**: ~$0.03
- **1000 images**: ~$30
- **Auto-scaling**: 0-10 containers

### Cloudflare R2 (Storage)

- **Storage**: $0.015/GB/month
- **Egress**: Free (huge savings!)
- **Operations**: Minimal cost
- **Estimated**: <$5/month

### Total Monthly Cost

- **Fixed**: $5 (Railway) + $0-5 (R2) = $5-10/month
- **Variable**: $0.03 per transformation
- **1000 transformations**: $35-40/month total

---

## 🚀 Deployment Steps

### 1. Deploy to Railway

```bash
git add .
git commit -m "Switch to minimal architecture"
git push origin main
```

Railway auto-deploys! ✨

### 2. Deploy to Modal

```bash
modal deploy modal_functions/sd_inference_complete.py
```

### 3. Verify

```bash
# Check Railway
curl https://playbooktv-interior-design-ai-production.up.railway.app/health

# Check Modal
modal app list
# Should show: modomo-sd-inference
```

---

## ✅ Benefits of Minimal Architecture

1. **Cost**: Railway Hobby plan works ($5 vs $20+)
2. **Speed**: Fast deploys (5 min vs 15+ min)
3. **Scalability**: Modal auto-scales GPU independently
4. **Maintenance**: Simpler Railway backend
5. **Reliability**: GPU failures don't affect API
6. **Development**: Faster iteration (deploy API without GPU deps)

---

## 📊 Performance Targets

| Metric | Target | Actual |
|--------|--------|--------|
| API response time | <200ms | TBD |
| Job creation | <500ms | TBD |
| WebSocket latency | <100ms | TBD |
| Total processing | <15s | TBD |
| Success rate | >92% | TBD |
| Cost per image | <$0.05 | $0.03 ✅ |

---

## 🔄 Migration from Old API

**Old Endpoint**: `/analyze` (uses YOLO + SAM2 on Railway)
**New Endpoint**: `/transform/submit` (uses Modal)

**Migration Plan**:
1. Deploy new `/transform` endpoints
2. Update mobile app to use `/transform`
3. Keep `/analyze` for legacy (optional)
4. Or: Move `/analyze` to Modal too

**Backward Compatibility**:
- Option A: Keep old API on Railway (requires hybrid requirements)
- Option B: Deprecate old API (recommended)
- Option C: Move old API to Modal function

---

## 🎯 Next Steps

1. ✅ Railway minimal setup complete
2. ⏳ Build Modal service interface
3. ⏳ Build new FastAPI endpoints
4. ⏳ Build WebSocket manager
5. ⏳ Build Modal complete processing function
6. ⏳ Test end-to-end
7. ⏳ Deploy to production

---

**Status**: Ready to build remaining services
**Architecture**: Minimal Railway + Modal GPU
**Cost**: ~$35-40/month for 1000 transformations
