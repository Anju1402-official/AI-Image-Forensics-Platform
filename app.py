import streamlit as st
import google.generativeai as genai
import cv2
import numpy as np
from PIL import Image, ImageChops
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Image Forensics Lab", layout="wide")

# --- CUSTOM CSS FOR UI/UX ---
st.markdown("""
    <style>
    .reportview-container { background: #f0f2f6; }
    .main-header { font-size: 2.5rem; color: #1E3A8A; font-weight: bold; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR: CONFIGURATION ---
st.sidebar.title("🛠️ Analysis Settings")
api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")
ela_quality = st.sidebar.slider("ELA Resave Quality", 10, 100, 90)

# --- FORENSIC FUNCTIONS ---

def perform_ela(image, quality):
    """
    Error Level Analysis: Resaves image at lower quality and finds the difference.
    """
    temp_filename = "temp_ela.jpg"
    image.save(temp_filename, 'JPEG', quality=quality)
    resaved_image = Image.open(temp_filename)
    
    # Calculate absolute difference
    ela_image = ImageChops.difference(image.convert("RGB"), resaved_image)
    
    # Scale the difference to make it visible
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0: max_diff = 1
    scale = 255.0 / max_diff
    
    ela_image = ImageChops.constant(ela_image, scale)
    os.remove(temp_filename)
    return ela_image

def get_ai_prediction(image, key):
    """
    Uses Gemini-Pro-Vision to analyze if the image is AI generated.
    """
    if not key:
        return "Please provide API Key", 0
    
    genai.configure(api_key=key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = """
    Act as a digital forensics expert. Analyze this image and determine if it is 
    AI-generated or a real photograph. Provide a one-sentence verdict starting 
    with 'Verdict:' and a confidence percentage. Look for GAN artifacts, 
    diffusion patterns, and lighting inconsistencies.
    """
    
    try:
        response = model.generate_content([prompt, image])
        return response.text
    except Exception as e:
        return f"Error: {str(e)}", 0

# --- MAIN UI ---
st.markdown('<p class="main-header">🕵️ AI Image Forensics Lab</p>', unsafe_allow_html=True)
st.write("Upload an image to perform deep semantic analysis and forensic error-level checks.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load Image
    img = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(img, use_container_width=True)
    
    with col2:
        st.subheader("ELA Forensic Map")
        ela_img = perform_ela(img, ela_quality)
        st.image(ela_img, use_container_width=True)
        st.caption("Bright areas in ELA indicate potential modifications or AI artifacts.")

    # --- EXECUTE AI DETECTION ---
    if st.button("🚀 Run AI Forensic Scan"):
        with st.spinner("Analyzing pixels and semantic patterns..."):
            verdict = get_ai_prediction(img, api_key)
            
            st.divider()
            st.subheader("🔍 Forensic Report")
            st.info(verdict)
            
            # Additional logic for a "Confidence Bar"
            if "Confidence:" in verdict:
                try:
                    conf = int(''.join(filter(str.isdigit, verdict.split("Confidence:")[-1])))
                    st.progress(conf/100)
                except: pass

# --- UI IMPROVEMENT SUGGESTION ---
st.sidebar.markdown("""
---
**Pro Mentor Tip:**
Use the 'ELA Forensic Map' to look for edges. In real photos, edges usually have higher error levels than flat surfaces. If a flat surface is "bright" in ELA, it was likely edited or synthesized.
""")
