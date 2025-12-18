import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageChops
import io

# --- PAGE CONFIG ---
st.set_page_config(page_title="Forensics Lab", layout="wide")

st.title("🕵️ AI Image Forensics Lab")

# --- FORENSIC FUNCTIONS ---

def get_ela(image_pil, quality=90):
    """Calculates Error Level Analysis."""
    buf = io.BytesIO()
    image_pil.convert("RGB").save(buf, format="JPEG", quality=quality)
    resaved_img = Image.open(io.BytesIO(buf.getvalue()))
    
    ela = ImageChops.difference(image_pil.convert("RGB"), resaved_img)
    extrema = ela.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0/max_diff if max_diff != 0 else 1
    return ImageChops.constant(ela, scale)

def get_sobel_map(img_cv):
    """Detects edges to find AI-generated structural inconsistencies."""
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobelx, sobely)
    return np.uint8(sobel_combined)

# --- UPLOADER LOGIC ---
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # PREVENT ERROR: Read the file once into bytes
    input_bytes = uploaded_file.read()
    
    # Convert for PIL
    img_pil = Image.open(io.BytesIO(input_bytes))
    
    # Convert for OpenCV
    file_bytes = np.frombuffer(input_bytes, np.uint8)
    img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # UI DISPLAY
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        # Using use_column_width for compatibility across Streamlit versions
        st.image(img_pil, use_column_width=True)
        
    with col2:
        st.subheader("ELA (Compression Analysis)")
        ela_img = get_ela(img_pil)
        st.image(ela_img, use_column_width=True)

    st.divider()

    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Sobel Edge Map")
        sobel_img = get_sobel_map(img_cv)
        st.image(sobel_img, use_column_width=True)
        
    with col4:
        st.subheader("Final Forensic Result")
        # Logic: If ELA average brightness is high, it's likely AI
        score = np.array(ela_img).mean()
        if score > 18:
            st.error(f"🚩 High Probability of AI/Edit (Score: {score:.2f})")
            st.write("The high ELA noise suggests the image was not captured by a standard physical sensor.")
        else:
            st.success(f"✅ Likely Authentic (Score: {score:.2f})")
            st.write("The noise levels are consistent with natural photographic sensors.")
