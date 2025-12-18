import streamlit as st
import numpy as np
from PIL import Image, ImageFilter
from skimage import feature, color

st.set_page_config(page_title="AI Image Forensics", layout="centered")
st.title("🕵️ AI Image Forensics – Real vs AI Generated")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

def analyze_image(img_pil):
    # Convert to grayscale
    gray = img_pil.convert("L")
    gray_np = np.array(gray, dtype=np.float32) / 255.0

    # Edge detection
    edges = feature.canny(gray_np, sigma=2, low_threshold=0.05, high_threshold=0.15)
    edge_density = np.mean(edges)

    # Laplacian noise approximation using PIL
    laplacian = np.array(gray.filter(ImageFilter.FIND_EDGES), dtype=float)
    noise_score = laplacian.var()

    # Heuristic scoring
    ai_score = 0
    if edge_density < 0.02:
        ai_score += 1
    if noise_score < 0.002:  # small threshold, normalized
        ai_score += 1

    confidence = min(95, 50 + ai_score * 25)

    if ai_score >= 2:
        return "AI Generated", confidence, edges
    else:
        return "Real Image", confidence, edges

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    label, confidence, edges = analyze_image(image)

    st.subheader("Uploaded Image")
    st.image(image, use_column_width=True)

    st.subheader("Forensic Edge Map")
    st.image((edges * 255).astype("uint8"), use_column_width=True)

    st.subheader("Prediction")
    if label == "AI Generated":
        st.error(f"⚠️ {label} ({confidence}% confidence)")
    else:
        st.success(f"✅ {label} ({confidence}% confidence)")
