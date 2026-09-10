import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Page configuration
st.set_page_config(page_title="Chest X-Ray Classifier", layout="centered")
st.title("Chest X-Ray Pneumonia Classifier")
st.write("Upload a chest X-ray image to check for signs of Pneumonia.")

# 2. Load the trained model safely
@st.cache_resource
def load_model():
    # Matches the exact configuration from your training notebook
    return tf.keras.models.load_model("chest_xray_model.keras")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

# 3. File uploader element
uploaded_file = st.file_uploader("Choose a Chest X-Ray image (JPG/PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-Ray Image", use_container_width=True)
    
    with st.spinner("Analyzing image..."):
        # Preprocessing matching your ImageDataGenerator specifications:
        # Grayscale, Resized to 224x224, Rescaled by 1/255.0, Batch dimension added.
        img_gray = image.convert("L")
        img_resized = img_gray.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_tensor = np.expand_dims(img_array, axis=(0, -1)) # Shape becomes (1, 224, 224, 1)
        
        # 4. Model Prediction
        prediction = model.predict(img_tensor)[0][0]
        
        # Using the optimal threshold calculated in your notebook validation phase
        threshold = 0.50 
        
        st.subheader("Results:")
        if prediction > threshold:
            st.error(f"**PNEUMONIA DETECTED** (Probability: {prediction:.2%})")
        else:
            st.success(f"**NORMAL** (Probability of Pneumonia: {prediction:.2%})")

st.info("**Disclaimer:** This tool is for educational and prototyping purposes only. It is not a substitute for professional clinical diagnosis.")
