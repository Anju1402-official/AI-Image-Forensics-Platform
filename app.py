import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageChops
import io

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Forensics Lab", layout="wide")
st.title("🕵️ AI Image Forensics Lab (Pro)")

def get_ela(image_pil, quality=90):
    """Calculates ELA and returns a safe uint8 NumPy array."""
    original = image_pil.convert("RGB")
    buf = io.BytesIO()
    original.save(buf, format="JPEG", quality=quality)
    resaved = Image.open(io.BytesIO(buf.getvalue()))
    diff = ImageChops.difference(original, resaved)
    extrema = diff.getextrema()
    max_diff = max([ex[1] if isinstance(ex, tuple) else ex for ex in extrema])
    if max_diff == 0: max_diff = 1
    scale = 255.0 / max_diff
    enhanced_diff = diff.point(lambda p: p * scale).convert("L")
    return np.array(enhanced_diff).astype(np.uint8)

def scan_metadata(input_bytes):
    """Scans raw bytes for AI generator signatures."""
    signatures = [b"DALL-E", b"Midjourney", b"Adobe Firefly", b"Stable Diffusion", b"Software: AI"]
    found = []
    for sig in signatures:
        if sig in input_bytes:
            found.append(sig.decode())
    return found

# --- UPLOADER ---
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    input_bytes = uploaded_file.read()
    raw_img = Image.open(io.BytesIO(input_bytes)).convert("RGB")
    
    # 1. Pixel Analysis (ELA)
    ela_map = get_ela(raw_img)
    ela_score = np.mean(ela_map) 

    # 2. Header Analysis (Metadata)
    ai_tags = scan_metadata(input_bytes)

    # --- UI LAYOUT ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(np.array(raw_img), use_column_width=True)
    with col2:
        st.subheader("ELA Forensic Map")
        st.image(ela_map, use_column_width=True, clamp=True)

    st.divider()

    # --- MULTI-FACTOR VERDICT ---
    st.header("🔍 Forensic Verdict")
    
    # Logic: AI if Tags found OR ELA score is very high
    is_ai = len(ai_tags) > 0 or ela_score > 12.0
    
    if is_ai:
        st.error("### Verdict: AI-GENERATED / MODIFIED")
        if ai_tags:
            st.info(f"**Digital Fingerprint Found:** {', '.join(ai_tags)}")
        st.write(f"**Forensic Score:** {ela_score:.2f}")
        st.warning("Reasoning: High-frequency inconsistencies or specific AI software signatures were detected in the file data.")
    else:
        st.success("### Verdict: LIKELY REAL PHOTOGRAPH")
        st.write(f"**Forensic Score:** {ela_score:.2f}")
        st.info("Reasoning: No known AI metadata found and compression noise appears natural.")

    st.sidebar.markdown(f"""
    **Stats for Report:**
    - Mean ELA: {ela_score:.2f}
    - AI Tags Found: {len(ai_tags)}
    """)
