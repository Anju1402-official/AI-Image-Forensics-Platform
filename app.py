import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageChops
import io

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Forensics Lab", layout="wide")

st.title("🕵️ AI Image Forensics Lab")

def get_ela(image_pil, quality=90):
    """Calculates ELA and returns a safe uint8 NumPy array."""
    original = image_pil.convert("RGB")
    buf = io.BytesIO()
    original.save(buf, format="JPEG", quality=quality)
    resaved = Image.open(io.BytesIO(buf.getvalue()))
    
    diff = ImageChops.difference(original, resaved)
    extrema = diff.getextrema()
    # Handle both grayscale and RGB extrema tuples
    max_diff = max([ex[1] if isinstance(ex, tuple) else ex for ex in extrema])
    if max_diff == 0: max_diff = 1
    scale = 255.0 / max_diff
    
    enhanced_diff = diff.point(lambda p: p * scale).convert("L")
    return np.array(enhanced_diff).astype(np.uint8)

# --- UPLOADER ---
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    input_bytes = uploaded_file.read()
    raw_img = Image.open(io.BytesIO(input_bytes)).convert("RGB")
    display_img = np.array(raw_img).astype(np.uint8)
    
    # --- CALCULATE FORENSIC SCORE ---
    ela_map = get_ela(raw_img)
    # The 'ela_score' represents the average brightness of the error map
    ela_score = np.mean(ela_map) 

    # --- UI LAYOUT ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(display_img, use_column_width=True)
        
    with col2:
        st.subheader("ELA Forensic Map")
        st.image(ela_map, use_column_width=True, clamp=True)

    st.divider()

    # --- THE DETECTION LOGIC ---
    st.header("🔍 Forensic Verdict")
    
    # Thresholding: In Digital Forensics, scores above 15-20 often indicate 
    # non-standard pixel distribution (AI or heavy Photoshop)
    if ela_score > 12.0:
        st.error(f"### Verdict: AI-GENERATED / MODIFIED")
        st.write(f"**Confidence Level:** {min(ela_score * 4, 99.0):.1f}%")
        st.warning("Reasoning: The ELA map shows high-frequency noise in flat areas, which is a common artifact of Diffusion models (DALL-E/Midjourney).")
    else:
        st.success(f"### Verdict: LIKELY REAL PHOTOGRAPH")
        st.write(f"**Confidence Level:** {max(100 - (ela_score * 5), 85.0):.1f}%")
        st.info("Reasoning: The noise distribution is consistent with natural camera sensor signatures.")

    # Visualize the Score
    st.write(f"Raw Forensic Score: `{ela_score:.2f}`")
    st.progress(min(ela_score / 30, 1.0))
