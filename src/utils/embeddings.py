"""CLIP image embedding utilities"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


def embed_image_pil(img: Image.Image, device=None, clip_model=None, clip_processor=None):
    """Return L2-normalized CLIP image embedding (np.array shape [1, dim])."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if clip_model is None or clip_processor is None:
        clip_name = "openai/clip-vit-base-patch32"
        if clip_processor is None:
            clip_processor = CLIPProcessor.from_pretrained(clip_name)
        if clip_model is None:
            clip_model = CLIPModel.from_pretrained(clip_name).to(device).eval()
    
    with torch.no_grad():
        inputs = clip_processor(images=img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        emb = clip_model.get_image_features(**inputs)
        emb = F.normalize(emb, p=2, dim=-1)
    return emb.detach().cpu().numpy()
