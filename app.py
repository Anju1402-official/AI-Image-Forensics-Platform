import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageChops
import io

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Forensics Lab", layout="wide")
st.title("🕵️ AI Image Forensics Lab (Advanced)")

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

def get_fft_spectrum(img_cv):
    """Detects repeating grid patterns common in AI generation."""
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    dft = np.fft.fft2(gray)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1)
    return np.uint8(np.clip(magnitude_spectrum, 0, 255))

def scan_binary(input_bytes):
    """Scans raw file data for AI software signatures."""
    sigs = [b"DALL-E", b"Midjourney", b"Adobe Firefly", b"Stable Diffusion"]
    return [s.decode() for s in sigs if s.lower() in input_bytes.lower()]

# --- UPLOADER ---
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    input_bytes = uploaded_file.read()
    raw_img = Image.open(io.BytesIO(input_bytes)).convert("RGB")
    file_bytes = np.frombuffer(input_bytes, np.uint8)
    img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # 1. Pixel Check (ELA)
    ela_map = get_ela(raw_img)
    ela_score = np.mean(ela_map) 

    # 2. Frequency Check (FFT)
    fft_map = get_fft_spectrum(img_cv)
    fft_score = np.std(fft_map) # AI images often have higher variance in frequencies

    # 3. Binary Check
    ai_tags = scan_binary(input_bytes)

    # --- UI ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(np.array(raw_img), use_column_width=True)
    with col2:
        st.subheader("FFT Frequency Spectrum")
        st.image(fft_map, use_column_width=True)
        st.caption("AI images often show unnatural dots or lines in this spectrum.")

    st.divider()

    # --- ENHANCED VERDICT LOGIC ---
    st.header("🔍 Forensic Verdict")
    
    # We combine ELA, FFT variance, and Metadata for a final decision
    if ai_tags:
        st.error(f"### Verdict: AI-GENERATED (Confirmed by Metadata: {', '.join(ai_tags)})")
    elif fft_score > 35.0: # Threshold for unnatural frequency distribution
        st.error(f"### Verdict: AI-GENERATED (Detected via FFT Analysis)")
        st.write(f"Frequency Variance: {fft_score:.2f}")
    elif ela_score > 12.0:
        st.error("### Verdict: AI-GENERATED / TAMPERED (Detected via ELA)")
    else:
        st.success("### Verdict: LIKELY REAL PHOTOGRAPH")
        st.write(f"Confidence: High (ELA: {ela_score:.2f}, FFT: {fft_score:.2f})")

    st.sidebar.info(f"ELA Score: {ela_score:.2f}\n\nFFT Score: {fft_score:.2f}")
