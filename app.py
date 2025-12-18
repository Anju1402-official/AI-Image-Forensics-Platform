import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageChops
import io
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Forensics Lab", layout="wide")

st.title("🕵️ AI Image Forensics Lab")

# --- FORENSIC FUNCTIONS ---

def get_ela(image_pil, quality=90):
    """Calculates Error Level Analysis with Type-Safe scaling."""
    # Ensure image is RGB
    image_pil = image_pil.convert("RGB")
    
    # Resave and reload
    buf = io.BytesIO()
    image_pil.save(buf, format="JPEG", quality=quality)
    resaved_img = Image.open(io.BytesIO(buf.getvalue()))
    
    # Calculate difference
    ela = ImageChops.difference(image_pil, resaved_img)
    
    # Find max difference for scaling
    extrema = ela.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    
    # FIX: Ensure scale is an INTEGER to prevent PIL TypeError
    scale = int(255 / max_diff)
    
    # Apply constant scale
    return ImageChops.multiply(ela, ImageChops.constant(ela, scale))

def get_sobel_map(img_cv):
    """Detects edges to find structural inconsistencies."""
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobelx, sobely)
    return np.uint8(np.absolute(sobel_combined))

# --- UPLOADER LOGIC ---
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read bytes once to avoid stream exhaustion
    input_bytes = uploaded_file.read()
    
    # Load for PIL (ELA)
    img_pil = Image.open(io.BytesIO(input_bytes))
    
    # Load for OpenCV (Sobel)
    file_bytes = np.frombuffer(input_bytes, np.uint8)
    img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(img_pil, use_container_width=True)
        
    with col2:
        st.subheader("ELA Forensic Map")
        # Try-Except block as a mentor-recommended safety measure
        try:
            ela_img = get_ela(img_pil)
            st.image(ela_img, use_container_width=True)
        except Exception as e:
            st.error(f"ELA Error: {e}")

    st.divider()

    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Sobel Edge Map")
        sobel_img = get_sobel_map(img_cv)
        st.image(sobel_img, use_container_width=True)
        
    with col4:
        st.subheader("Final Forensic Result")
        # Heuristic assessment based on ELA brightness
        ela_array = np.array(get_ela(img_pil))
        score = np.mean(ela_array)
        
        if score > 15:
            st.error(f"🚩 High Probability of AI/Edit (Score: {score:.2f})")
            st.write("Analysis: Found unnatural compression artifacts often seen in AI generation.")
        else:
            st.success(f"✅ Likely Authentic (Score: {score:.2f})")
            st.write("Analysis: Pixel distribution matches standard digital sensor behavior.")
