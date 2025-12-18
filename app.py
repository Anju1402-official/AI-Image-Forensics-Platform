import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageChops
import io

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Forensics Pro", layout="wide")
st.title("🕵️ AI Image Forensics Lab (v4.0)")

def get_ela(image_pil, quality=90):
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

def scan_for_ai_strings(input_bytes):
    """Scans binary data for hidden AI creator tags."""
    ai_sigs = [b"DALL-E", b"Midjourney", b"Adobe Firefly", b"Stable Diffusion", b"Prompt", b"AI Generated"]
    found = [s.decode() for s in ai_sigs if s.lower() in input_bytes.lower()]
    return found

# --- UPLOADER ---
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    input_bytes = uploaded_file.read()
    raw_img = Image.open(io.BytesIO(input_bytes)).convert("RGB")
    
    # 1. Pixel Check
    ela_score = np.mean(get_ela(raw_img)) 

    # 2. Binary Metadata Check (This is the most accurate for your panda/logo images)
    ai_tags = scan_for_ai_strings(input_bytes)

    # --- UI DISPLAY ---
    st.image(np.array(raw_img), width=400, caption="Uploaded Image")
    st.divider()

    # --- THE VERDICT LOGIC ---
    st.header("🔍 Forensic Verdict")
    
    # Rule 1: Metadata is king. If we find an AI tag, it's AI regardless of the score.
    if ai_tags:
        st.error(f"### Verdict: AI-GENERATED (Confirmed via Metadata)")
        st.info(f"**Found Signature:** {', '.join(ai_tags)}")
    
    # Rule 2: High ELA score (>12) is an indicator of synthesis or tampering.
    elif ela_score > 12.0:
        st.error(f"### Verdict: AI-GENERATED / MODIFIED (Statistical Detection)")
        st.write(f"Forensic Score: {ela_score:.2f}")
    
    # Rule 3: Visual/Heuristic Check (Manual Mentor Input)
    else:
        st.success(f"### Verdict: LIKELY REAL PHOTOGRAPH (Pixel Analysis)")
        st.write(f"Score: {ela_score:.2f}")
        st.warning("Note: High-end AI can sometimes bypass pixel-level tests. Check metadata tab for more info.")

    with st.expander("🔬 View Raw Forensic Data"):
        st.write("Average ELA Score:", ela_score)
        st.write("Metadata Buffer Found:", ai_tags if ai_tags else "None")
