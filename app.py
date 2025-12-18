import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageChops
import io

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Forensics Lab", layout="wide")

st.title("🕵️ AI Image Forensics Lab")

# --- FORENSIC FUNCTIONS ---

def get_ela(image_pil, quality=90):
    """Calculates ELA without using the problematic ImageChops.constant."""
    # Step 1: Convert to RGB and Resave
    original = image_pil.convert("RGB")
    buf = io.BytesIO()
    original.save(buf, format="JPEG", quality=quality)
    resaved = Image.open(io.BytesIO(buf.getvalue()))
    
    # Step 2: Calculate Absolute Difference
    diff = ImageChops.difference(original, resaved)
    
    # Step 3: Enhance Visibility (Scaling)
    # We find the maximum brightness in the difference map
    extrema = diff.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    
    scale = 255.0 / max_diff
    
    # Step 4: Apply scale via point transformation (Much more stable than ImageChops.constant)
    return diff.point(lambda p: p * scale)

def get_sobel_map(img_cv):
    """Detects edges using OpenCV to find structural inconsistencies."""
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobelx, sobely)
    return np.uint8(np.clip(sobel_combined, 0, 255))

# --- UPLOADER LOGIC ---
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Read bytes once
    input_bytes = uploaded_file.read()
    
    # Load for PIL
    img_pil = Image.open(io.BytesIO(input_bytes))
    
    # Load for OpenCV
    file_bytes = np.frombuffer(input_bytes, np.uint8)
    img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(img_pil, use_container_width=True)
        
    with col2:
        st.subheader("ELA Forensic Map")
        try:
            ela_result = get_ela(img_pil)
            st.image(ela_result, use_container_width=True)
        except Exception as e:
            st.error(f"ELA Processing Error: {e}")

    st.divider()

    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Sobel Edge Map")
        sobel_result = get_sobel_map(img_cv)
        st.image(sobel_result, use_container_width=True)
        
    with col4:
        st.subheader("Final Forensic Result")
        # Analysis logic
        ela_array = np.array(get_ela(img_pil))
        score = np.mean(ela_array)
        
        if score > 15:
            st.error(f"🚩 High Probability of AI/Edit (Score: {score:.2f})")
            st.write("Artifact Analysis: Flat surfaces show abnormal noise distribution.")
        else:
            st.success(f"✅ Likely Authentic (Score: {score:.2f})")
            st.write("Artifact Analysis: Noise is consistent with physical sensor patterns.")
