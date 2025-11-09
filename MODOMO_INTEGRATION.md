# Modomo App Integration Guide

**Last Updated**: 2025-11-08
**Purpose**: Connect PlaybookTV Interior Design AI to your Modomo app

---

## 🎯 Complete Integration Flow

```
Modomo App (Frontend)
    ↓ Upload Image
PlaybookTV API (FastAPI on Paperspace)
    ↓ Process with AI
YOLO (Detect 294 objects) + Style Classifier (9 styles)
    ↓ Return Results
Modomo App (Display to user)
```

---

## 📦 What You Have

Based on your files:

1. ✅ **Database**: `database_metadata.duckdb` (18 MB)
2. ✅ **Phase 1 Model**: `models_best_interior_model.pth` (130 MB)
3. ✅ **API Server Code**: Just created in `api/main.py`
4. ✅ **Your API Keys**: Already have all of them
5. ✅ **Paperspace Instance**: Running at your URL

---

## 🚀 Quick Setup (30 Minutes)

### Step 1: Upload to Paperspace (10 min)

**In your Paperspace terminal** (https://n8mwcw7u7e.clg07azjl.paperspacegradient.com):

```bash
# Navigate to notebooks folder
cd /notebooks

# Upload your project
# Option A: If code is on GitHub
git clone https://github.com/thePlaybookTV/playbooktv-interior-design-ai.git

# Option B: Upload manually
# - Zip project on your Mac
# - Upload via Paperspace file browser
# - Unzip: unzip project.zip
```

### Step 2: Upload Database (5 min)

Upload `database_metadata.duckdb` from your Mac to Paperspace:
- Use Paperspace file upload
- Put it in `/notebooks/playbooktv-interior-design-ai/`

### Step 3: Set Up API (10 min)

```bash
cd playbooktv-interior-design-ai/api

# Install dependencies
pip install fastapi uvicorn python-multipart python-dotenv pillow torch torchvision ultralytics

# Create .env file
cp .env.example .env
nano .env

# Edit .env and add your API keys:
# (Already have them - just copy/paste from your snippet)
ROBOFLOW_API_KEY=qgdh7zxmWdGi2N8xHsVF
KAGGLE_USERNAME=pearlisa
KAGGLE_KEY=ec87fb61ac251c5076abb1418da07c5b
HUGGINGFACE_TOKEN=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
UNSPLASH_ACCESS_KEY=_emTInR1snz-qBJYltXwot6fBsfZkJfEjLsEGn-s7U0
PEXELS_API_KEY=Ncl1eB8fS4A8qPj5yViznt7FlJ36k672yoOihJvXg9ZwqbZMHCChSBhj
```

### Step 4: Start API (5 min)

```bash
# Start the server
python main.py
```

You should see:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 5: Get Public URL

**Option A: Use ngrok** (Easiest):
```bash
# In another Paperspace terminal
pip install pyngrok

python << EOF
from pyngrok import ngrok
public_url = ngrok.connect(8000)
print(f"\n🌐 Your API URL: {public_url}")
print("\nUse this URL in your Modomo app!")
input("Press Enter to stop...")
EOF
```

**Option B: Paperspace URL**:
Your Paperspace URL is accessible at:
`https://n8mwcw7u7e.clg07azjl.paperspacegradient.com:8000`

---

## 💻 Modomo App Integration

### Add to your Modomo app:

**1. Create API service file** (`src/services/interiorAI.js`):

```javascript
const API_URL = 'https://your-ngrok-url.ngrok.io'; // Or Paperspace URL

export async function analyzeInteriorImage(imageFile) {
  const formData = new FormData();
  formData.append('file', imageFile);

  const response = await fetch(`${API_URL}/analyze`, {
    method: 'POST',
    body: formData
  });

  if (!response.ok) {
    throw new Error(`Analysis failed: ${response.statusText}`);
  }

  return await response.json();
}
```

**2. Use in your component**:

```javascript
import { analyzeInteriorImage } from './services/interiorAI';

function ImageUploader() {
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleUpload = async (event) => {
    const file = event.target.files[0];
    setLoading(true);

    try {
      const analysis = await analyzeInteriorImage(file);
      setResults(analysis);

      // You now have:
      // - analysis.style.style (e.g., "modern")
      // - analysis.style.confidence (e.g., 0.78)
      // - analysis.detections (array of detected furniture)
      // - analysis.detection_count

      console.log('Detected items:', analysis.detections);
      console.log('Style:', analysis.style.style);

    } catch (error) {
      console.error('Analysis failed:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <input type="file" onChange={handleUpload} accept="image/*" />

      {loading && <p>Analyzing...</p>}

      {results && (
        <div>
          <h3>Style: {results.style.style}</h3>
          <p>Confidence: {(results.style.confidence * 100).toFixed(1)}%</p>

          <h4>Detected Items ({results.detection_count}):</h4>
          <ul>
            {results.detections.map((item, i) => (
              <li key={i}>{item.item_type} - {(item.confidence * 100).toFixed(1)}%</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
```

**3. Full example in `api/client_example.js`**

---

## 🎨 What You Get from the API

### Complete Analysis Response

```json
{
  "detections": [
    {
      "item_type": "sectional_sofa",
      "confidence": 0.89,
      "bbox": [120.5, 200.3, 450.2, 380.7],
      "area_percentage": 25.3
    },
    {
      "item_type": "coffee_table",
      "confidence": 0.85,
      "bbox": [300.1, 350.2, 500.5, 450.8],
      "area_percentage": 12.5
    },
    {
      "item_type": "accent_chair",
      "confidence": 0.82,
      "bbox": [500.0, 180.0, 650.0, 400.0],
      "area_percentage": 15.2
    }
  ],
  "detection_count": 3,
  "style": {
    "style": "modern",
    "confidence": 0.78,
    "all_probabilities": {
      "modern": 0.78,
      "contemporary": 0.12,
      "minimalist": 0.05,
      "scandinavian": 0.03,
      "traditional": 0.02,
      "industrial": 0.00,
      "bohemian": 0.00,
      "mid_century_modern": 0.00,
      "rustic": 0.00
    }
  },
  "processing_time_ms": 234.5
}
```

### Use Cases in Modomo

**1. Furniture Recognition**:
```javascript
// Show detected furniture to user
const furniture = results.detections.map(d => d.item_type);
console.log('Found:', furniture);
// ["sectional_sofa", "coffee_table", "accent_chair"]
```

**2. Style Recommendations**:
```javascript
// Recommend items based on detected style
if (results.style.style === 'modern') {
  showModernFurnitureRecommendations();
}
```

**3. Smart Search**:
```javascript
// Search for similar items
const searchQuery = results.detections[0].item_type;
searchDatabase(searchQuery); // Search for "sectional_sofa"
```

**4. Shopping Cart**:
```javascript
// Add detected items to shopping list
results.detections.forEach(item => {
  addToShoppingList({
    type: item.item_type,
    style: results.style.style
  });
});
```

---

## ⚡ API Endpoints Reference

### `POST /analyze`
**Complete analysis** - Detection + Style (Recommended)
- **Input**: Image file
- **Output**: Full analysis
- **Speed**: ~200-300ms

### `POST /detect`
**Detection only** - Faster
- **Input**: Image file
- **Output**: Just detections
- **Speed**: ~100-150ms

### `POST /classify/style`
**Style only**
- **Input**: Image file
- **Output**: Just style classification
- **Speed**: ~100-150ms

### `GET /health`
**Health check**
- Verify API is running

### `GET /models/info`
**Model information**
- See what models are loaded

---

## 🔧 Configuration Options

### Different Model Scenarios

**Scenario 1: Using Phase 1 Model (Quick Start)**
```bash
# In api/.env
YOLO_MODEL_PATH=../models_best_interior_model.pth  # Phase 1 model
# Note: Phase 1 has 14 generic classes, lower accuracy
```

**Scenario 2: After Phase 2 Training (Best)**
```bash
# In api/.env
YOLO_MODEL_PATH=../phase2_outputs/yolo_training_runs/finetune_294_classes/weights/best.pt
EFFICIENTNET_PATH=../best_efficientnet_style_classifier.pth
RESNET_PATH=../best_resnet_style_classifier.pth
VIT_PATH=../best_vit_style_classifier.pth
# Note: 294 classes, 70%+ accuracy
```

**Scenario 3: Pre-trained Models (If you have them)**
```bash
# Upload pre-trained models to Paperspace
# Then point to them in .env
```

---

## 📊 Performance Expectations

### With Phase 1 Models:
- **Object Detection**: 14 generic classes
- **Style Accuracy**: ~54%
- **Speed**: Fast (~100ms)

### With Phase 2 Models:
- **Object Detection**: 294 specific classes
- **Style Accuracy**: ~70%
- **Speed**: Moderate (~200ms)

### Response Times:
- **CPU**: 1-2 seconds per image
- **GPU** (Paperspace): 200-300ms per image

---

## 🐛 Troubleshooting

### API Not Starting

```bash
# Check if port is in use
lsof -i :8000

# Use different port
PORT=8080 python main.py
```

### Models Not Loading

```bash
# Check model paths
python << EOF
import os
print("YOLO:", os.path.exists("./models/yolo_best.pt"))
print("EfficientNet:", os.path.exists("./models/best_efficientnet_style_classifier.pth"))
EOF
```

### CORS Errors from Modomo

Update `api/main.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-modomo-app.com"],  # Your actual domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Connection Refused from Modomo

1. Check API is running: `curl http://localhost:8000/health`
2. Check ngrok tunnel is active
3. Update Modomo app with correct URL

---

## 🎯 Complete Deployment Checklist

- [ ] Upload code to Paperspace
- [ ] Upload `database_metadata.duckdb`
- [ ] Create `api/.env` with API keys
- [ ] Install dependencies: `pip install -r api/requirements.txt`
- [ ] Configure model paths in `.env`
- [ ] Start API: `python api/main.py`
- [ ] Get public URL (ngrok or Paperspace)
- [ ] Test API: `curl http://your-url/health`
- [ ] Update Modomo app with API URL
- [ ] Test from Modomo app
- [ ] Deploy! 🎉

---

## 📝 Next Steps

1. **Now**: Get API running on Paperspace
2. **Tomorrow**: Integrate with Modomo app
3. **Next Week**: Run Phase 2 training for better models
4. **Next Month**: Deploy to production cloud (AWS/GCP)

---

## 🆘 Need Help?

**Documentation**:
- [API_DEPLOYMENT_GUIDE.md](API_DEPLOYMENT_GUIDE.md) - Full deployment guide
- [api/client_example.js](api/client_example.js) - Integration examples
- [START_HERE.md](START_HERE.md) - Overall setup guide

**Test API**:
```bash
# Health check
curl https://your-api-url/health

# Test with image
curl -X POST "https://your-api-url/analyze" \
  -F "file=@test_image.jpg"
```

---

## 💡 Pro Tips

1. **Start simple**: Get API working with Phase 1 models first
2. **Train later**: Run Phase 2 training overnight for better models
3. **Monitor costs**: Paperspace charges by the hour - stop when not using
4. **Use ngrok**: Easiest way to get public URL quickly
5. **Add auth**: Use API keys for production

---

**You're all set!** Follow the steps above and you'll have your Modomo app connected to the AI in about 30 minutes! 🚀
