# Railway Deployment Options

## Problem

The existing API (`api/main.py`) uses heavy ML models (YOLO, SAM2, Phase1 classifier) which makes the Docker image very large and Railway deployment expensive.

## Solution Options

---

### Option 1: Minimal API (Recommended ✅)

**Use**: `requirements-railway-minimal.txt`

**Strategy**:
- Railway = lightweight API only (job queue, WebSocket, Modal communication)
- Modal = ALL heavy processing (YOLO, SAM2, SD, control generation)

**Pros**:
- ✅ Smallest Docker image (~500MB vs 3GB+)
- ✅ Fastest Railway deploys (<5 min vs 15+ min)
- ✅ Cheapest Railway plan (Hobby $5/month works)
- ✅ Scalable (Modal auto-scales GPU, Railway scales API)
- ✅ Clean separation of concerns

**Cons**:
- ❌ Breaks existing `/analyze` endpoint (needs migration)
- ❌ All processing on Modal (slight latency increase)

**Architecture**:
```
Mobile App
    ↓
Railway API (minimal)
  - Receive upload
  - Create Redis job
  - Send to Modal
  - WebSocket updates
    ↓
Modal (all GPU work)
  - YOLO + SAM2 detection (if needed)
  - Generate control images
  - SD 1.5 + ControlNet
  - Return result
```

**Cost per transformation**: ~£0.03-0.05

---

### Option 2: Hybrid API (Backward Compatible)

**Use**: `requirements-railway.txt`

**Strategy**:
- Railway = lightweight + depth generation
- Modal = SD inference only
- Keep existing `/analyze` endpoint on Modal too

**Pros**:
- ✅ Smaller than full (800MB vs 3GB+)
- ✅ Keeps depth generation on Railway
- ✅ Can migrate `/analyze` gradually

**Cons**:
- ❌ Still includes PyTorch (~700MB)
- ❌ Railway needs more resources
- ❌ Slower deploys (~10 min)

**Architecture**:
```
Mobile App
    ↓
Railway API (medium)
  - Receive upload
  - Generate depth map (DPT)
  - Generate edges (OpenCV)
  - Send to Modal for SD
    ↓
Modal (SD only)
  - SD 1.5 + ControlNet
  - Return result
```

**Cost per transformation**: ~£0.03-0.05

---

### Option 3: Full API (Legacy)

**Use**: `requirements.txt`

**Strategy**:
- Railway = ALL models (YOLO, SAM2, Phase1, SD on Modal)
- Keep existing API fully functional

**Pros**:
- ✅ Existing `/analyze` endpoint works
- ✅ Backward compatible

**Cons**:
- ❌ Huge Docker image (3GB+)
- ❌ Railway Pro required ($20/month minimum)
- ❌ Slow deploys (15-20 min)
- ❌ High memory usage (2GB+ RAM)
- ❌ Expensive to scale

**Architecture**:
```
Mobile App
    ↓
Railway API (full)
  - YOLO + SAM2 detection
  - Phase1 classifier
  - Generate control images
  - Send to Modal for SD
    ↓
Modal (SD only)
  - SD 1.5 + ControlNet
  - Return result
```

**Cost per transformation**: £0.03-0.05 (SD) + £0.01-0.02 (Railway GPU equivalent)

---

## Recommendation: Option 1 (Minimal)

**Why?**

1. **Cost**: Hobby plan works ($5/month)
2. **Performance**: Fast deploys, fast response
3. **Scalability**: Modal handles all heavy lifting
4. **Simplicity**: Clean architecture

**Migration Plan**:

1. **New endpoint**: `/transform/submit` (already planned)
   - Uses Modal for everything
   - Async with job queue

2. **Old endpoint**: `/analyze` (deprecate or migrate)
   - Option A: Keep it, deploy on Modal too
   - Option B: Remove it (mobile app uses `/transform` instead)

3. **Gradual rollout**:
   - Phase 1: Add `/transform` endpoints (new)
   - Phase 2: Update mobile app to use `/transform`
   - Phase 3: Deprecate `/analyze` (optional)

---

## Decision Matrix

| Feature | Minimal | Hybrid | Full |
|---------|---------|--------|------|
| Docker image size | 500MB | 800MB | 3GB+ |
| Deploy time | 5 min | 10 min | 15-20 min |
| Railway cost | $5/mo | $10-15/mo | $20+/mo |
| Memory usage | 512MB | 1GB | 2GB+ |
| `/analyze` works | No* | Yes | Yes |
| `/transform` works | Yes | Yes | Yes |
| Scalability | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |

*Can be deployed on Modal instead

---

## Implementation Steps (Minimal)

### 1. Update Dockerfile

```dockerfile
# Use minimal requirements
COPY requirements-railway-minimal.txt .
RUN pip install --no-cache-dir -r requirements-railway-minimal.txt
```

### 2. Create New API Endpoints

```python
# api/main.py (simplified)

@app.post("/transform/submit")
async def submit_transformation(file: UploadFile, style: str):
    """
    New transformation endpoint - all processing on Modal
    """
    # 1. Upload to R2
    # 2. Create Redis job
    # 3. Submit to Modal
    # 4. Return job_id

@app.get("/transform/status/{job_id}")
async def get_status(job_id: str):
    """
    Check transformation status
    """
    # Query Redis job queue

@app.websocket("/ws/transform/{job_id}")
async def transform_websocket(websocket: WebSocket, job_id: str):
    """
    Real-time progress updates
    """
    # Stream from Redis PubSub
```

### 3. Deploy Modal Function

```python
# modal_functions/sd_inference.py

@stub.function(gpu="T4")
def transform_room_complete(image_url, style):
    """
    Complete transformation pipeline on Modal:
    1. Download image from R2
    2. Run YOLO + SAM2 (if needed)
    3. Generate control images
    4. Run SD 1.5 + ControlNet
    5. Upload result to R2
    6. Return result URL
    """
    # Implementation...
```

### 4. Update Mobile App

```typescript
// Use new endpoint
const result = await api.submitTransformation(imageUri, style)
const jobId = result.job_id

// Subscribe to WebSocket for progress
const ws = new WebSocket(`wss://api.../ws/transform/${jobId}`)
ws.onmessage = (update) => {
  // Update UI with progress
}
```

---

## File Structure

```
requirements.txt                    # Full (for local development)
requirements-railway.txt            # Hybrid (with depth generation)
requirements-railway-minimal.txt    # Minimal (recommended for Railway)
```

**Update Dockerfile to use**:
```dockerfile
COPY requirements-railway-minimal.txt .
RUN pip install -r requirements-railway-minimal.txt
```

---

## Questions to Answer

1. **Do you need the existing `/analyze` endpoint?**
   - If yes → Use hybrid or full
   - If no → Use minimal (recommended)

2. **Is the mobile app already using `/analyze`?**
   - If yes → Need migration plan
   - If no → Build new from scratch with `/transform`

3. **What's your budget?**
   - <$10/month → Minimal only option
   - $20+/month → Hybrid or full work

---

## My Recommendation

**Use Option 1 (Minimal)** because:

1. You're building for SD transformation (new feature)
2. Mobile app isn't built yet (no legacy endpoint)
3. Budget-conscious (£500/month total = £470 for Modal)
4. Scalability matters for growth

**Next Step**: I'll build the new API with minimal requirements and move all heavy processing to Modal.

---

**Decision**: Which option do you want to use?

- [ ] **Option 1: Minimal** (recommended)
- [ ] **Option 2: Hybrid**
- [ ] **Option 3: Full**
