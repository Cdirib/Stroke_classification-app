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


st.markdown(
    """
    <div style='
        border: 2px solid #4A90E2;
        background-color: #E8F0FE;
        padding: 15px;
        border-radius: 20px;
        color: #1A237E;
        text-align: center;
        font-size: 30px;
        font-weight: bold;
    '>
        🧠 Stroke Types Prediction
    </div>
    """,
    unsafe_allow_html=True
)
# st.write("Upload a stroke image to automatically predict its classification: Ischemic, Hemorrhagic, or Transient Ischemic Attack.")
st.markdown(
    """
    <p style='
        border: 2px solid #4A90E2;
        background-color: #E8F0FE;
        padding: 15px;
        border-radius: 20px;
        color: #1A237E;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
    '>
        Upload a stroke image to automatically predict its classification: 
        <strong>Ischemic</strong>, <strong>Hemorrhagic</strong>, or <strong>Transient Ischemic Attack</strong>.
    </p>
    """,
    unsafe_allow_html=True
)
# Image upload
uploaded_file = st.file_uploader("📤 Choose an image...", type=["jpg", "jpeg", "png"])

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
    st.write(f"### Prediction: **{labels[class_idx]}**")
    st.write(f"Confidence: {confidence:.2f}")
