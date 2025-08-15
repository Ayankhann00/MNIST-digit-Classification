import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2 as cv
from streamlit_drawable_canvas import st_canvas

# Set page title and favicon
st.set_page_config(page_title="MNIST Digit Predictor", page_icon="✏️")

# Title and description
st.title("MNIST Digit Predictor")
st.markdown("Draw a digit (0-9) on the canvas below, and the model will predict the number.")

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mnist_cnn_model.h5")

model = load_model()

# Create a canvas for drawing
st.subheader("Draw a digit here:")
canvas_result = st_canvas(
    fill_color="black",  # Black background to match MNIST
    stroke_width=20,     # Thick stroke for clear digits
    stroke_color="white", # White stroke to mimic MNIST digits
    background_color="black",
    height=280,          # 280x280 canvas
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Function to preprocess the drawn image
def preprocess_image(image):
    # Convert to grayscale
    img = Image.fromarray(image).convert("L")
    # Resize to 28x28 pixels
    img = img.resize((28, 28), Image.LANCZOS)
    # Convert to numpy array and normalize
    img_array = np.array(img, dtype="float32") / 255.0
    # Invert colors (white digit on black background, like MNIST)
    img_array = 1.0 - img_array
    # Reshape for model input
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array

# Predict button
if st.button("Predict"):
    if canvas_result.image_data is not None:
        # Get the drawn image
        image = canvas_result.image_data
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Make prediction
        prediction = model.predict(processed_image)
        predicted_digit = np.argmax(prediction)
        probabilities = prediction[0]

        # Display the processed 28x28 image
        st.subheader("Processed Image (28x28):")
        st.image(processed_image.reshape(28, 28), clamp=True, caption="Input to Model", width=100)

        # Display prediction
        st.subheader("Prediction:")
        st.write(f"**Predicted Digit: {predicted_digit}**")
        
        # Display probability distribution
        st.subheader("Prediction Probabilities:")
        prob_dict = {str(i): float(prob) for i, prob in enumerate(probabilities)}
        st.bar_chart(prob_dict)
    else:
        st.error("Please draw a digit on the canvas before predicting.")

# Instructions
st.markdown("""
### Instructions:
1. Draw a digit (0-9) on the canvas using your mouse or touch.
2. Click the **Predict** button to see the model's prediction.
3. The processed 28x28 image and prediction probabilities will be displayed.
""")
