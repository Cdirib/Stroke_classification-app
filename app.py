import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model
model = load_model("stroke-prediction_model.h5")

# Set image dimensions (based on your training setup)
IMG_SIZE = (224, 224)

# Define label mapping
labels = ['Normal', 'Stroke']

# Styled Title
st.markdown(
    """
    <div style='
        border: 2px solid #4A90E2;
        background-color: #E8F0FE;
        padding: 15px;
        border-radius: 10px;
        color: #1A237E;
        text-align: center;
        font-size: 30px;
        font-weight: bold;
        margin-bottom: 10px;
    '>
        🧠 Stroke Types Prediction
    </div>
    """,
    unsafe_allow_html=True
)

# Styled Subtitle
st.markdown(
    """
    <p style='
        background-color: #F5F5F5;
        padding: 10px 15px;
        border-left: 5px solid #4A90E2;
        border-radius: 5px;
        font-size: 20px;
        color: green;
        margin-bottom: 20px;
    '>
        Upload a stroke image to automatically predict its classification: 
        <strong>Ischemic</strong>, <strong>Hemorrhagic</strong>, or <strong>Transient Ischemic Attack</strong>.
    </p>
    """,
    unsafe_allow_html=True
)

# Styled file uploader section
st.markdown(
    """
    <div style='
        border: 2px dashed #4A90E2;
        padding: 20px;
        border-radius: 10px;
        background-color: #FAFAFA;
        margin-top: 10px;
        margin-bottom: 20px;
    '>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("📤 Choose an image...", type=["jpg", "jpeg", "png"])

st.markdown("</div>", unsafe_allow_html=True)

# Process the uploaded image
if uploaded_file is not None:
    # Read and display image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(image, channels="BGR", caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    image_resized = cv2.resize(image, IMG_SIZE)
    image_array = img_to_array(image_resized)
    image_array = np.expand_dims(image_array, axis=0) / 255.0

    # Predict
    prediction = model.predict(image_array)[0]
    class_idx = np.argmax(prediction)
    confidence = prediction[class_idx]

    # Display results
    st.markdown(f"<h3 style='color: #1A237E;'>Prediction: <strong>{labels[class_idx]}</strong></h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:18px;'>Confidence: <strong>{confidence:.2f}</strong></p>", unsafe_allow_html=True)
