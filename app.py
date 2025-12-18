import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageChops
import io

# --- PAGE CONFIG ---
st.set_page_config(page_title="Advanced Forensics", layout="wide")
st.title("🕵️ AI Image Forensics Lab (v3.0)")

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
    return np.array(diff.point(lambda p: p * scale).convert("L")).astype(np.uint8)

def scan_binary_signatures(input_bytes):
    """Scans for hidden software tags in the file's binary header."""
    signatures = [b"DALL-E", b"Midjourney", b"Adobe Firefly", b"Stable Diffusion", b"kandinsky"]
    found = [sig.decode() for sig in signatures if sig.lower() in input_bytes.lower()]
    return found

# --- UPLOADER ---
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    input_bytes = uploaded_file.read()
    raw_img = Image.open(io.BytesIO(input_bytes)).convert("RGB")
    
    # 1. Pixel Check (ELA)
    ela_map = get_ela(raw_img)
    ela_score = np.mean(ela_map) 

    # 2. Binary Metadata Check
    ai_tags = scan_binary_signatures(input_bytes)

    # --- UI RENDERING ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(np.array(raw_img), use_column_width=True)
    with col2:
        st.subheader("ELA Forensic Map")
        st.image(ela_map, use_column_width=True)

    st.divider()

    # --- THE VERDICT LOGIC ---
    st.header("🔍 Forensic Verdict")
    
    # AI detection criteria:
    # - ELA Score > 10 (Statistical anomaly)
    # - Metadata contains AI tags (Absolute confirmation)
    
    if ai_tags:
        st.error(f"### Verdict: AI-GENERATED (Confirmed via Metadata)")
        st.write(f"**Found Signature:** {', '.join(ai_tags)}")
        st.info("Reasoning: The file's internal binary header contains tags from known AI generators.")
    elif ela_score > 10.0:
        st.error(f"### Verdict: AI-GENERATED / MODIFIED (High Probability)")
        st.write(f"Forensic Score: {ela_score:.2f}")
        st.warning("Reasoning: Pixel-level inconsistencies detected in the compression layer.")
    else:
        st.success(f"### Verdict: LIKELY REAL PHOTOGRAPH")
        st.write(f"Forensic Score: {ela_score:.2f}")
        st.info("Reasoning: No digital AI signatures found; compression noise is consistent with physical sensors.")

st.sidebar.markdown("""
### 👨‍🏫 Mentor Tips:
If ELA is failing (low score), it means the AI image is high-quality. 
**Next Step:** Use the **Metadata Check** to find hidden 'DALL-E' strings.
""")
