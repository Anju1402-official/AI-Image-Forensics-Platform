import torch
from PIL import Image
import numpy as np
from transformers import CLIPProcessor, CLIPModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model once
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

AI_PROMPT = "an AI generated image"
REAL_PROMPT = "a real photograph taken by a camera"

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        text=[AI_PROMPT, REAL_PROMPT],
        images=image,
        return_tensors="pt",
        padding=True
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]

    ai_score = float(logits[0])
    real_score = float(logits[1])

    if ai_score > real_score:
        label = "AI-Generated Image"
        confidence = ai_score * 100
        trust = real_score * 100
    else:
        label = "Real Image"
        confidence = real_score * 100
        trust = ai_score * 100

    return {
        "label": label,
        "confidence": round(confidence, 2),
        "trust_score": round(trust, 2),
        "raw_scores": {
            "ai": round(ai_score, 4),
            "real": round(real_score, 4)
        }
    }
