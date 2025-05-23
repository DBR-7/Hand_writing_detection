import streamlit as st
import joblib
import numpy as np
from PIL import Image, ImageOps

# Load the trained model
model = joblib.load("model.pkl")

st.title("MNIST Digit Classifier (3 vs 5)")
st.write("Upload an image of a handwritten digit (3 or 5). The model will classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

def preprocess_image(image):
    # Convert to grayscale, resize to 28x28, invert, and flatten
    image = ImageOps.grayscale(image)
    image = image.resize((28, 28))
    image = ImageOps.invert(image)
    image_array = np.array(image).reshape(1, -1)
    return image_array

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    processed = preprocess_image(image)
    prediction = model.predict(processed)[0]
    confidence = model.predict_proba(processed)[0][prediction]

    label = "5" if prediction == 1 else "3"
    st.write(f"### Prediction: {label}")
    st.write(f"Confidence: {confidence:.2%}")