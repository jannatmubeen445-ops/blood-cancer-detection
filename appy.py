import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Model load karo
model = tf.keras.models.load_model("blood_cancer_cnn.keras")

# Classes
class_names = ["Benign", "Early", "Pre", "Pro"]

st.title("Blood Cancer Detection App")
st.write("Upload a blood cell image for prediction.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Image ko model ke input size mein convert karo
    img = image.resize((224, 224))

    # NumPy array
    img_array = np.array(img)

    # Batch dimension add karo
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    st.success(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")
    