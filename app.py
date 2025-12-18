import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from skimage import io, color, feature

# --------------------------
# Load your ML model
# --------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("your_model_path_here")  # replace with your actual model
    return model

model = load_model()

# --------------------------
# Helper functions
# --------------------------
def preprocess_image(image: Image.Image):
    # Convert to array
    img_array = np.array(image)
    # Resize to model input size (change as per your model)
    img_array = cv2.resize(img_array, (224, 224))
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def analyze_image(image: Image.Image):
    # Example: run ML prediction
    processed = preprocess_image(image)
    prediction = model.predict(processed)
    
    # Example: simple feature extraction
    gray = color.rgb2gray(np.array(image))
    edges = feature.canny(gray)
    
    return prediction, edges

# --------------------------
# Streamlit frontend
# --------------------------
st.title("AI Image Forensics Platform")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Analyze"):
        with st.spinner("Analyzing..."):
            pred, edges = analyze_image(image)
            st.success("Analysis complete!")
            
            st.subheader("ML Prediction Output")
            st.write(pred)
            
            st.subheader("Edge Detection Preview")
            st.image(edges, caption="Canny Edges", use_column_width=True)
