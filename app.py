import streamlit as st
import google.generativeai as genai
import PIL.Image
import io
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="Pro AI Forensics", layout="wide")
st.title("🕵️ Pro AI Image Forensics (Deep Learning)")

# --- SETUP API ---
# Get your free key at https://aistudio.google.com/app/apikey
st.sidebar.title("Configuration")
api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")

def analyze_with_ai(image_pil, key):
    genai.configure(api_key=key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = """
    Analyze this image as a digital forensics expert. 
    1. Determine if it is AI-generated or a real photograph.
    2. Look for anatomical errors, unnatural lighting, or GAN artifacts.
    3. Start your response with 'VERDICT: AI' or 'VERDICT: REAL'.
    4. Provide a confidence percentage.
    """
    
    response = model.generate_content([prompt, image_pil])
    return response.text

# --- UPLOADER ---
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = PIL.Image.open(uploaded_file)
    st.image(img, caption="Target Image", width=500)
    
    if st.button("🚀 Run Deep Analysis"):
        if not api_key:
            st.error("Please enter your API Key in the sidebar!")
        else:
            with st.spinner("Expert AI is inspecting pixels and semantic patterns..."):
                try:
                    result = analyze_with_ai(img, api_key)
                    
                    st.divider()
                    st.header("🔍 Forensic Report")
                    
                    if "VERDICT: AI" in result.upper():
                        st.error(result)
                    else:
                        st.success(result)
                except Exception as e:
                    st.error(f"Error: {e}")

st.sidebar.markdown("""
### 👨‍🏫 Mentor's Note:
Classical ELA fails because modern AI 'masks' its noise. 
Deep Learning APIs look at the **content** (like weird fingers or perfect lighting) 
to catch AI where math fails.
""")
