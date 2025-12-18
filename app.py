import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="AI vs Real Image Detector", layout="centered")
st.title("🧠 AI vs Real Image Detection")

# Load your pre-trained model (must be placed in same folder)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("ai_vs_real_model.h5")

model = load_model()

uploaded_file = st.file_uploader("Upload an image (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess for model
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    # Model inference
    pred = model.predict(img_batch)[0][0]
    # Adjust threshold if model outputs probabilities
    label = "AI‑Generated" if pred > 0.5 else "Real"
    confidence = float(pred if pred > 0.5 else 1 - pred) * 100

    # Show results
    if label == "AI‑Generated":
        st.error(f"🚨 {label} ({confidence:.2f}% confidence)")
    else:
        st.success(f"✅ {label} ({confidence:.2f}% confidence)")
