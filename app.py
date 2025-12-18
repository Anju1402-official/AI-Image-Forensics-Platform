import streamlit as st
from skimage import io, color, feature
import numpy as np

st.title("AI Image Forensics - Edge Detection")

# Upload image
uploaded_file = st.file_uploader("Choose an image (jpg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    image = io.imread(uploaded_file)

    # Convert to grayscale
    gray = color.rgb2gray(image)

    # Canny edge detection
    edges = feature.canny(gray, sigma=2, low_threshold=0.05, high_threshold=0.15)

    # Display original and edges side by side
    st.subheader("Original Image")
    st.image(image, use_column_width=True)

    st.subheader("Detected Edges")
    st.image((edges * 255).astype("uint8"), use_column_width=True)
