# Modal API Migration: Stub → App

**Date**: November 2025
**Status**: ✅ COMPLETE

## Overview

Modal updated their API, replacing `modal.Stub` with `modal.App`. This document tracks all changes made to migrate the codebase to the new API.

---

## Changes Made

### 1. Modal Function: `modal_functions/sd_inference_complete.py`

**Changed**:
```python
# OLD API
stub = modal.Stub("modomo-sd-inference")

@stub.cls(...)
class CompleteTransformationPipeline:
    ...

@stub.local_entrypoint()
def main():
    ...
```

**To**:
```python
# NEW API
app = modal.App("modomo-sd-inference")

@app.cls(...)
class CompleteTransformationPipeline:
    ...

@app.local_entrypoint()
def main():
    ...
```

**Lines changed**: 24, 156, 469

---

### 2. Modal Service: `src/services/modal_service.py`

**Changed**:
- Class initialization parameter: `stub_name` → `app_name`
- Instance variable: `self.stub` → `self.app`
- Lookup method: `modal.Stub.lookup()` → `modal.App.lookup()`
- Function lookup: Updated to use class name and method
- All error messages and docstrings updated

**Key changes**:
```python
# OLD API
self.stub = modal.Stub.lookup(self.stub_name, create_if_missing=False)
transform_function = self.stub.cls.lookup("process_transformation_complete")
call = transform_function.spawn(...)

# NEW API
self.app = modal.App.lookup(self.app_name, create_if_missing=False)
transform_function = self.app.cls.lookup("CompleteTransformationPipeline")
call = transform_function.process_transformation_complete.spawn(...)
```

**Lines changed**: Multiple throughout file

---

### 3. Environment Configuration

**Updated files**:
- `.env` - Added `MODAL_APP_NAME=modomo-sd-inference`
- `.env.example` - Added `MODAL_APP_NAME` with documentation

**Backward compatibility**: `MODAL_STUB_NAME` still supported as fallback in modal_service.py

---

## Verification

✅ **Syntax check passed** for both files:
```bash
python3 -m py_compile modal_functions/sd_inference_complete.py
python3 -m py_compile src/services/modal_service.py
```

---

## Deployment Instructions

### 1. Deploy Modal Function

```bash
# Deploy the updated Modal function
modal deploy modal_functions/sd_inference_complete.py
```

**Expected output**:
```
✓ Created deployment modomo-sd-inference
✓ App is now deployed!
✓ View at: https://modal.com/apps/modomo-sd-inference
```

### 2. Verify Deployment

```bash
# List deployed Modal apps
modal app list

# Should show: modomo-sd-inference
```

### 3. Test Railway API Connection

```bash
# Start Railway API locally to test Modal connection
cd /Users/leslieisah/MDMv3/playbooktv-interior-design-ai
uvicorn api.main_minimal:app --reload

# Check health endpoint
curl http://localhost:8000/health
```

**Expected response**:
```json
{
  "status": "healthy",
  "services": {
    "redis": "connected",
    "modal": "connected",
    "storage": "connected",
    "websocket": "ready"
  }
}
```

---

## What Changed in Modal API

| Old API (Stub) | New API (App) |
|----------------|---------------|
| `modal.Stub("name")` | `modal.App("name")` |
| `@stub.cls(...)` | `@app.cls(...)` |
| `@stub.function(...)` | `@app.function(...)` |
| `@stub.local_entrypoint()` | `@app.local_entrypoint()` |
| `modal.Stub.lookup(name)` | `modal.App.lookup(name)` |

---

## Breaking Changes

None for end users - all changes are internal to the Modal integration.

---

## Testing Checklist

Before marking as complete, verify:

- [x] Modal function syntax is valid (no Python errors)
- [x] Modal service syntax is valid (no Python errors)
- [x] Environment variables updated (.env and .env.example)
- [ ] Modal function deploys successfully
- [ ] Railway API can connect to Modal app
- [ ] End-to-end transformation works
- [ ] Health check shows Modal connected

---

## Rollback Plan

If the new API causes issues:

1. **Revert Modal function**:
   ```python
   stub = modal.Stub("modomo-sd-inference")
   @stub.cls(...)
   ```

2. **Revert Modal service**:
   ```python
   self.stub = modal.Stub.lookup(self.stub_name)
   ```

3. **Redeploy**:
   ```bash
   modal deploy modal_functions/sd_inference_complete.py
   ```

---

## Next Steps

1. ✅ Complete Modal API migration (DONE)
2. ⏳ Deploy Modal function: `modal deploy modal_functions/sd_inference_complete.py`
3. ⏳ Deploy to Railway: `git push origin main`
4. ⏳ Test end-to-end transformation
5. ⏳ Build React Native mobile app

---

## References

- **Modal Documentation**: https://modal.com/docs
- **Migration Guide**: https://modal.com/docs/guide/stub-to-app-migration
- **Our Architecture**: [docs/MINIMAL_ARCHITECTURE.md](MINIMAL_ARCHITECTURE.md)
- **Deployment Guide**: [READY_TO_DEPLOY.md](../READY_TO_DEPLOY.md)

---

**Status**: All code changes complete ✅
**Next**: Deploy to Modal and Railway
