import streamlit as st
import numpy as np
import cv2
from PIL import Image
from skimage import feature

st.set_page_config(page_title="AI Image Forensics", layout="centered")
st.title("🕵️ AI Image Forensics – Real vs AI Generated")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

def analyze_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Edge density
    edges = feature.canny(gray, sigma=2, low_threshold=0.05, high_threshold=0.15)
    edge_density = np.mean(edges)

    # Noise level (Laplacian variance)
    noise_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Heuristic decision
    ai_score = 0

    if edge_density < 0.02:
        ai_score += 1
    if noise_score < 80:
        ai_score += 1

    confidence = min(95, 50 + ai_score * 25)

    if ai_score >= 2:
        return "AI Generated", confidence, edges
    else:
        return "Real Image", confidence, edges

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    label, confidence, edges = analyze_image(img_np)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.subheader("🔍 Forensic Edge Map")
    st.image((edges * 255).astype("uint8"))

    st.subheader("🧠 Prediction")
    if label == "AI Generated":
        st.error(f"⚠️ {label} ({confidence}% confidence)")
    else:
        st.success(f"✅ {label} ({confidence}% confidence)")
