import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from skimage import color, feature

st.set_page_config(page_title="AI Image Forensics", layout="centered")

# ------------------ MODEL ------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")  # adjust if needed

try:
    model = load_model()
    MODEL_OK = True
except:
    MODEL_OK = False

# ------------------ FUNCTIONS ------------------
def preprocess(img):
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def analyze(img):
    result = None
    if MODEL_OK:
        result = model.predict(preprocess(img))

    gray = color.rgb2gray(np.array(img))
    edges = feature.canny(gray)

    return result, edges

# ------------------ UI ------------------
st.title("🕵️ AI Image Forensics Platform")

file = st.file_uploader("Upload Image", ["jpg", "png", "jpeg"])

if file:
    image = Image.open(file).convert("RGB")
    st.image(image, use_column_width=True)

    if st.button("Analyze Image"):
        with st.spinner("Analyzing..."):
            pred, edges = analyze(image)

        st.success("Done")

        if pred is not None:
            st.subheader("Model Prediction")
            st.write(pred)

        st.subheader("Edge Detection")
        st.image(edges, clamp=True)
