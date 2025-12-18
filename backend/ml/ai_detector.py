from PIL import Image
import numpy as np
import random

def analyze_image(image_path):
    """
    Fake AI detector (placeholder).
    Later we will replace with a real pretrained model.
    """

    # Open image
    image = Image.open(image_path).convert("RGB")

    # Convert to numpy (simulating AI processing)
    image_array = np.array(image)

    # Fake logic (random for now)
    ai_probability = random.uniform(0.3, 0.9)

    if ai_probability > 0.6:
        label = "AI-Generated Image"
    else:
        label = "Real Image"

    confidence = round(ai_probability * 100, 2)
    trust_score = round(100 - confidence, 2)

    return {
        "label": label,
        "confidence": confidence,
        "trust_score": trust_score
    }
