import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageChops
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Forensics Lab", layout="wide")

st.title("🕵️ AI Image Forensics Lab")
st.write("Professional-grade forensic analysis using ELA and Noise Fingerprinting.")

# --- FORENSIC ENGINES ---

def get_ela(image_pil, quality=90):
    """Error Level Analysis (ELA) implementation."""
    temp_file = "temp_resave.jpg"
    # Convert to RGB to ensure JPEG compatibility
    image_pil = image_pil.convert("RGB")
    image_pil.save(temp_file, 'JPEG', quality=quality)
    resaved = Image.open(temp_file)
    
    ela = ImageChops.difference(image_pil, resaved)
    extrema = ela.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0/max_diff if max_diff != 0 else 1
    ela = ImageChops.constant(ela, scale)
    
    if os.path.exists(temp_file):
        os.remove(temp_file)
    return ela

def get_noise_map(img_cv):
    """Laplacian noise extraction for sensor fingerprinting."""
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    noise_map = np.uint8(np.absolute(laplacian))
    return noise_map

# --- UPLOADER ---
uploaded_file = st.file_uploader("Upload Image (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # 1. READ IMAGE FOR PIL (ELA)
    img_pil = Image.open(uploaded_file)
    
    # 2. READ IMAGE FOR OPENCV (Noise Map) - This fixes the TypeError
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_cv = cv2.imdecode(file_bytes, 1)

    # UI LAYOUT
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(img_pil, use_container_width=True)
        
    with col2:
        st.subheader("ELA Forensic Map")
        ela_img = get_ela(img_pil)
        st.image(ela_img, use_container_width=True)

    st.divider()
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Noise Fingerprint")
        noise_img = get_noise_map(img_cv)
        st.image(noise_img, use_container_width=True)
        
    with col4:
        st.subheader("Forensic Verdict")
        # Heuristic Assessment
        ela_stat = np.mean(np.array(ela_img))
        
        if ela_stat > 15:
            st.error(f"Verdict: AI GENERATED / TAMPERED (Score: {ela_stat:.2f})")
            st.write("High inconsistency in compression levels detected.")
        else:
            st.success(f"Verdict: LIKELY AUTHENTIC (Score: {ela_stat:.2f})")
            st.write("Compression levels appear consistent with natural capture.")

st.sidebar.info("Mentor Tip: Real images show edges in ELA; AI images show noise in flat areas.")
