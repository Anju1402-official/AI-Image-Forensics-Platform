import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageChops
import io

# --- PAGE CONFIG ---
st.set_page_config(page_title="Forensics Lab Pro", layout="wide")
st.title("🕵️ AI Image Forensics Lab (v2.0)")

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
    """Scans the raw binary of the image for AI software signatures."""
    # Common strings found in AI-generated or AI-edited image headers
    ai_signatures = [
        b"DALL-E", b"Midjourney", b"Adobe Firefly", b"Stable Diffusion", 
        b"kandinsky", b"creativelive", b"deepai", b"craiyon"
    ]
    found_tags = []
    # Convert to lowercase for broader matching
    lower_bytes = input_bytes.lower()
    for sig in ai_signatures:
        if sig.lower() in lower_bytes:
            found_tags.append(sig.decode())
    return found_tags

# --- UPLOADER ---
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Read raw bytes for metadata scan
    input_bytes = uploaded_file.read()
    raw_img = Image.open(io.BytesIO(input_bytes)).convert("RGB")
    
    # 1. Pixel Analysis (ELA)
    ela_map = get_ela(raw_img)
    ela_score = np.mean(ela_map) 

    # 2. Header Analysis (Metadata)
    found_ai_metadata = scan_metadata(input_bytes)

    # --- UI ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(np.array(raw_img), use_column_width=True)
    with col2:
        st.subheader("ELA Forensic Map")
        st.image(ela_map, use_column_width=True, clamp=True)

    st.divider()

    # --- FINAL VERDICT LOGIC ---
    st.header("🔍 Forensic Verdict")
    
    # If metadata is found, it's 100% AI regardless of ELA score
    if len(found_ai_metadata) > 0:
        st.error(f"### Verdict: AI-GENERATED (Confirmed via Metadata)")
        st.info(f"**Digital Signature Found:** {', '.join(found_ai_metadata)}")
        st.warning("Reasoning: The file header contains specific software tags used by AI image generators.")
    
    # Fallback to ELA if no metadata is found
    elif ela_score > 12.0:
        st.error(f"### Verdict: AI-GENERATED / MODIFIED (Statistical Analysis)")
        st.write(f"Forensic Score: {ela_score:.2f}")
        st.warning("Reasoning: High inconsistency in pixel compression levels detected.")
    
    else:
        st.success(f"### Verdict: LIKELY REAL PHOTOGRAPH")
        st.write(f"Forensic Score: {ela_score:.2f}")
        st.info("Reasoning: No AI metadata signatures found and pixel noise appears consistent.")

    st.sidebar.write(f"**Internal Log:**")
    st.sidebar.write(f"ELA Score: {ela_score:.2f}")
    st.sidebar.write(f"Metadata Hits: {len(found_ai_metadata)}")
