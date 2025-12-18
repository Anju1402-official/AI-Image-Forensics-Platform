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
    """Calculates ELA and returns a safe NumPy array."""
    # Ensure standard RGB
    original = image_pil.convert("RGB")
    
    # Resave/Reload to find compression difference
    buf = io.BytesIO()
    original.save(buf, format="JPEG", quality=quality)
    resaved = Image.open(io.BytesIO(buf.getvalue()))
    
    # Calculate Difference
    diff = ImageChops.difference(original, resaved)
    
    # Scale for visibility
    extrema = diff.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0: max_diff = 1
    scale = 255.0 / max_diff
    
    # Return as NumPy array to avoid Streamlit/Pillow metadata crashes
    enhanced_diff = diff.point(lambda p: p * scale)
    return np.array(enhanced_diff)

def get_sobel_map(img_cv):
    """Detects structural edges using OpenCV."""
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobelx, sobely)
    # Normalize to 0-255
    return np.uint8(np.clip(sobel_combined, 0, 255))

# --- UPLOADER ---
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Read file into memory once
    input_bytes = uploaded_file.read()
    
    # 1. Load for Original Display (Force conversion to RGB array)
    raw_img = Image.open(io.BytesIO(input_bytes)).convert("RGB")
    display_img = np.array(raw_img)
    
    # 2. Load for OpenCV
    file_bytes = np.frombuffer(input_bytes, np.uint8)
    img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # --- UI RENDERING ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        # Passing NumPy array is safer than passing PIL object
        st.image(display_img, use_container_width=True)
        
    with col2:
        st.subheader("ELA Forensic Map")
        try:
            ela_arr = get_ela(raw_img)
            st.image(ela_arr, use_container_width=True)
        except Exception as e:
            st.error(f"ELA Error: {e}")

    st.divider()

    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Sobel Edge Map")
        sobel_arr = get_sobel_map(img_cv)
        st.image(sobel_arr, use_container_width=True)
        
    with col4:
        st.subheader("Forensic Verdict")
        # Heuristic: Mean of the ELA map
        ela_score = np.mean(get_ela(raw_img))
        
        if ela_score > 15:
            st.error(f"🚩 High Probability of AI/Edit (Score: {ela_score:.2f})")
            st.write("**Analysis:** Unnatural noise levels detected in smooth regions.")
        else:
            st.success(f"✅ Likely Authentic (Score: {ela_score:.2f})")
            st.write("**Analysis:** Compression patterns match standard camera sensors.")
