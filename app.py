import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import os

st.set_page_config(page_title="MNIST Digit Predictor", page_icon="✏️")

st.title("MNIST Digit Predictor")
st.markdown("Draw a digit (0-9) on the canvas below, and the model will predict the number.")

def train_and_save_model():
    st.info("Training new MNIST model... please wait ⏳")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1)),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=2, verbose=1)
    model.save("mnist_cnn_model.keras")
    st.success("Model trained and saved ✅")
    return model

@st.cache_resource
def load_model():
    if not os.path.exists("mnist_cnn_model.keras"):
        return train_and_save_model()
    return tf.keras.models.load_model("mnist_cnn_model.keras")

model = load_model()

st.subheader("Draw a digit here:")
canvas_result = st_canvas(
    fill_color="black",  
    stroke_width=20,     
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

def preprocess_image(image):
    img = Image.fromarray(image).convert("L")
    img = img.resize((28, 28), Image.LANCZOS)
    img_array = np.array(img, dtype="float32") / 255.0
    img_array = 1.0 - img_array
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array

if st.button("Predict"):
    if canvas_result.image_data is not None:
        processed_image = preprocess_image(canvas_result.image_data)
        prediction = model.predict(processed_image)
        predicted_digit = np.argmax(prediction)
        probabilities = prediction[0]

        st.subheader("Processed Image (28x28):")
        st.image(processed_image.reshape(28, 28), clamp=True, caption="Input to Model", width=100)

        st.subheader("Prediction:")
        st.write(f"**Predicted Digit: {predicted_digit}**")

        st.subheader("Prediction Probabilities:")
        prob_dict = {str(i): float(prob) for i, prob in enumerate(probabilities)}
        st.bar_chart(prob_dict)
    else:
        st.error("Please draw a digit on the canvas before predicting.")
        
st.markdown("""
### Instructions:
1. Draw a digit (0-9) on the canvas using your mouse or touch.
2. Click the **Predict** button to see the model's prediction.
3. The processed 28x28 image and prediction probabilities will be displayed.
""")
