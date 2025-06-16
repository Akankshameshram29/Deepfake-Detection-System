import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

st.title("Deepfake Detection System")

# Load the model
model_path = "my_deepfake_detector.h5"
if os.path.exists(model_path):
    model = load_model(model_path)
    st.success("Model loaded successfully.")
else:
    st.error(f"Model file not found at: {model_path}")
    st.stop()

# Upload image
uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((128, 128))  # Adjust to your model's expected input size
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]  # Assuming binary classification
    label = "Fake 😈" if prediction > 0.5 else "Real 😇"


    st.write(f"### Prediction: {label}")
    st.write(f"Confidence: {prediction:.4f}")
