import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os
import cv2
import tempfile
from datetime import datetime

# ---------- Page Configuration ----------
st.set_page_config(
    page_title="Deepfake Detector",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown("""
    <style>
    /* Force background for main app area */
    .stApp {
        background-color: #0d1b2a !important;  /* Light gray-blue */
    }

    /* Optional: background inside main container (if needed) */
    .main .block-container {
        background-color: #f5f7fa !important;
        padding: 2rem;
        border-radius: 10px;
    }

    .title {
        font-size: 36px;
        color: #1c3f60;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0px;
    }

    .subtitle {
        text-align: center;
        font-size: 16px;
        color: #555;
        margin-top: 0px;
    }

    .uploadbox {
        background-color: #e8eef4;
        padding: 20px;
        border-radius: 10px;
        border: 2px dashed #aaa;
        text-align: center;
        color: #333;
    }

    .stButton>button {
        background-color: #0066cc;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
        transition: 0.3s;
    }

    .stButton>button:hover {
        background-color: #004a99;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown('<div class="title">🧠 Deepfake Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an image or video to check if it\'s AI-generated or authentic.</div>', unsafe_allow_html=True)

# ---------- Load Model ----------
model_path = "my_deepfake_detector.h5"
if os.path.exists("my_deepfake_detector.h5"):
    model = load_model("my_deepfake_detector.h5")
    st.success("✅ Deepfake Detection Model Loaded.")
else:
    st.error("❌ Model not found. Please check the model path.")
    st.stop()

# ---------- Setup Contribution Directory ----------
contrib_dir = "user_contributions"
os.makedirs(os.path.join(contrib_dir, "real"), exist_ok=True)
os.makedirs(os.path.join(contrib_dir, "fake"), exist_ok=True)

def save_user_input(file, label):
    ext = file.name.split('.')[-1]
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{label.lower()}_{timestamp}.{ext}"
    path = os.path.join(contrib_dir, label.lower(), filename)
    with open(path, "wb") as f:
        f.write(file.read())
    st.success(f"✅ Saved to {label} contributions for retraining.")

# ---------- Choose Type ----------
st.subheader("📂 Upload Your Media")
input_type = st.radio("Select Input Type", ("Image", "Video"))

# ---------- Image Upload ----------
if input_type == "Image":
    uploaded_image = st.file_uploader("📸 Upload Image (JPEG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        st.image(uploaded_image, caption="Preview", use_column_width=True)
        image = Image.open(uploaded_image).convert("RGB")
        img = image.resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0][0]
        label = "Fake 😈" if prediction > 0.5 else "Real 😇"
        confidence = prediction if prediction > 0.5 else 1 - prediction

        st.markdown(f"### 🧾 Prediction: {label}")
        st.markdown(f"📊 Confidence:** {confidence*100:.2f}%")

        with st.expander("ℹ What does this mean?"):
            if prediction > 0.5:
                st.write("The model believes this is likely a deepfake or AI-manipulated image.")
            else:
                st.write("The model considers this a real human face without signs of manipulation.")

        if st.checkbox("Allow use of this image for improving our model"):
            if st.button("📤 Contribute Image"):
                save_user_input(uploaded_image, "Fake" if prediction > 0.5 else "Real")

# ---------- Video Upload ----------
if input_type == "Video":
    uploaded_video = st.file_uploader("🎥 Upload Video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        st.video(tfile.name)
        st.info("⏳ Analyzing video...")

        cap = cv2.VideoCapture(tfile.name)
        frame_count = 0
        predictions = []
        frame_skip = 10

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(frame, (128, 128))
            input_tensor = np.expand_dims(np.array(resized) / 255.0, axis=0)
            pred = model.predict(input_tensor)[0][0]
            predictions.append(pred)

        cap.release()

        if len(predictions) > 0:
            avg_pred = np.mean(predictions)
            label = "Fake 😈" if avg_pred > 0.5 else "Real 😇"
            confidence = avg_pred if avg_pred > 0.5 else 1 - avg_pred
            st.markdown(f"### 🎬 Video Prediction: {label}")
            st.markdown(f"📊 Confidence:** {confidence*100:.2f}%")

            if st.checkbox("Allow use of this video for model improvement"):
                if st.button("📥 Contribute Video"):
                    save_user_input(uploaded_video, "Fake" if avg_pred > 0.5 else "Real")
        else:
            st.warning("⚠ Could not process enough frames. Try a longer or clearer video.")

# ---------- Footer ----------
st.markdown("---")
st.markdown(
    "<center style='color:#aaa'>© 2025 Akanksha Meshram | Built with ❤ using Streamlit</center>",
    unsafe_allow_html=True,
)