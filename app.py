 import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from keras.models import load_model

# Load model (cached so it doesn't reload every time)
@st.cache_resource
def load_my_model():
    return load_model("mnist_model.h5")

model = load_my_model()

st.title("Digit Recognizer")

uploaded_file = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Convert image to grayscale
    image = Image.open(uploaded_file).convert('L')

    # Invert image (important for MNIST)
    image = ImageOps.invert(image)

    # Resize to 28x28
    image = image.resize((28, 28))

    # Convert to array
    img_array = np.array(image)

    # Normalize
    img_array = img_array / 255.0

    # Reshape for model
    img_array = img_array.reshape(1, 28, 28)

    # Predict
    prediction = model.predict(img_array)
    result = np.argmax(prediction)
    confidence = np.max(prediction)

    # Display
    st.image(image, caption="Processed Image (28x28)")
    st.write(f"### Predicted Digit: {result}")
    st.write(f"### Confidence: {confidence:.2f}")
