import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageChops
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Forensics Tool", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stAlert { border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- CORE FORENSIC ENGINE ---

def get_ela(image, quality=90):
    """Calculates Error Level Analysis."""
    temp_file = "temp_resave.jpg"
    image.save(temp_file, 'JPEG', quality=quality)
    resaved = Image.open(temp_file)
    
    ela = ImageChops.difference(image.convert("RGB"), resaved)
    extrema = ela.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0/max_diff if max_diff != 0 else 1
    ela = ImageChops.constant(ela, scale)
    
    if os.path.exists(temp_file):
        os.remove(temp_file)
    return ela

def get_noise_map(image):
    """Extracts high-frequency noise using Laplacian filter."""
    img_array = np.array(image.convert('L')) # Grayscale
    laplacian = cv2.Laplacian(img_array, cv2.CV_64F)
    noise_map = np.uint8(np.absolute(laplacian))
    return noise_map

# --- UI LAYOUT ---
st.title("🕵️ Digital Image Forensics Lab")
st.write("Analyze pixel-level inconsistencies to detect AI generation or tampering.")

uploaded_file = st.file_uploader("Upload an image for forensic analysis", type=["jpg", "png", "jpeg"])

if uploaded_file:
    original_img = Image.open(uploaded_file)
    
    # Grid Layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🖼️ Original Image")
        st.image(original_img, use_container_width=True)
        
    with col2:
        st.subheader("🧪 ELA Analysis")
        ela_result = get_ela(original_img)
        st.image(ela_result, use_container_width=True)
        st.caption("AI-generated images often show uniform 'bright' noise. Real photos have noise concentrated on edges.")

    st.divider()

    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("📡 Noise Fingerprint")
        noise_result = get_noise_map(original_img)
        st.image(noise_result, use_container_width=True, clamp=True)
        st.caption("Highlights 'sensor noise'. Inconsistencies suggest non-camera origins.")
        
    with col4:
        st.subheader("📋 Forensic Assessment")
        # Logic-based heuristic (Mentor Tip: This is a simplified detection rule)
        ela_data = np.array(ela_result)
        mean_ela = np.mean(ela_data)
        
        if mean_ela > 20:
            st.error("Verdict: HIGH Probability of AI/Tampering")
            st.write(f"Digital Signature Score: {mean_ela:.2f}")
            st.warning("Analysis: The image lacks standard compression patterns found in natural photography.")
        else:
            st.success("Verdict: Likely Real/Original")
            st.write(f"Digital Signature Score: {mean_ela:.2f}")
            st.info("Analysis: Compression patterns are consistent with standard digital sensors.")

st.sidebar.markdown("""
### 👨‍🏫 Mentor Instructions
1. **Upload** a JPG file for best results.
2. **Interpret ELA:** If the whole image looks 'sparkly' or bright, it's a sign of AI synthesis.
3. **Interpret Noise:** Real photos have a grainy texture; AI images often look 'smeared' in this view.
""")
