import streamlit as st
import requests
from PIL import Image
import io

# -------------------------------
# CONFIG
# -------------------------------
BACKEND_URL = "http://127.0.0.1:5000/upload"

st.set_page_config(
    page_title="AI Image Forensics Platform",
    page_icon="🕵️‍♂️",
    layout="centered"
)

# -------------------------------
# STYLES (Beach / Sea Aesthetic 🌊)
# -------------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(180deg, #a1c4fd, #c2e9fb);
}
.block-container {
    padding-top: 2rem;
}
.result-box {
    background-color: rgba(255,255,255,0.85);
    padding: 20px;
    border-radius: 15px;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# TITLE
# -------------------------------
st.title("🕵️ AI Image Forensics & Prompt Intelligence")
st.subheader("AI vs Real Image Detection")
st.write("Upload **any image** to check whether it is AI-generated or a real photograph.")

# -------------------------------
# IMAGE UPLOAD
# -------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload an image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Analyze Image"):
        with st.spinner("Analyzing image... please wait"):
            files = {
                "image": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type
                )
            }

            try:
                response = requests.post(BACKEND_URL, files=files)
                data = response.json()

                analysis = data.get("analysis", {})

                label = analysis.get("label", "Unknown")
                confidence = analysis.get("confidence", 0)
                trust = analysis.get("trust_score", 0)

                st.markdown('<div class="result-box">', unsafe_allow_html=True)

                # -------------------------------
                # UNCERTAINTY LOGIC
                # -------------------------------
                if confidence < 60:
                    st.warning("⚠️ Model is unsure about this prediction.")

                st.markdown(f"### 🧠 Prediction: **{label}**")
                st.markdown(f"**Confidence:** {confidence}%")
                st.markdown(f"**Opposite Trust Score:** {trust}%")

                st.progress(int(confidence))

                st.markdown('</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error("❌ Failed to connect to backend.")
                st.code(str(e))
