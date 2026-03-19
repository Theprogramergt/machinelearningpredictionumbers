import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from PIL import Image

st.title("🔢 MNIST Digit Recognizer")
st.write("Upload a handwritten digit image and the model will predict it!")

# ─── Load & train model (cached so it only runs once) ───────────────────────
@st.cache_resource
def load_model():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train / 255.0
    x_test  = x_test  / 255.0

    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test  = x_test.reshape(-1, 28, 28, 1)

    model = models.Sequential([
        tf.keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(x_train, y_train, epochs=3, verbose=0)
    return model, x_test, y_test

with st.spinner("Training model on MNIST... (first load only, ~1 min)"):
    model, x_test, y_test = load_model()

st.success("Model ready!")

# ─── Section 1: Test with a dataset image ───────────────────────────────────
st.header("1. Test with a Dataset Image")
index = st.slider("Pick an image index from test set", 0, len(x_test) - 1, 0)

image      = x_test[index]
true_label = y_test[index]
prediction = model.predict(image.reshape(1, 28, 28, 1), verbose=0)
predicted  = np.argmax(prediction)
confidence = np.max(prediction) * 100

col1, col2 = st.columns(2)
with col1:
    st.image(image.reshape(28, 28), width=200, caption="Test Image", clamp=True)
with col2:
    st.metric("Predicted", predicted)
    st.metric("True Label", true_label)
    st.metric("Confidence", f"{confidence:.1f}%")
    if predicted == true_label:
        st.success("✅ Correct!")
    else:
        st.error("❌ Wrong prediction")

# ─── Section 2: Upload your own image ───────────────────────────────────────
st.header("2. Upload Your Own Digit Image")
uploaded = st.file_uploader("Upload a handwritten digit (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("L")
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0

    if np.mean(img_array) > 0.5:      # auto-invert white background
        img_array = 1 - img_array

    img_array = (img_array > 0.3).astype(float)

    prediction = model.predict(img_array.reshape(1, 28, 28, 1), verbose=0)
    predicted  = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    col1, col2 = st.columns(2)
    with col1:
        st.image(img_array.reshape(28, 28), width=200, caption="Your Image", clamp=True)
    with col2:
        st.metric("Predicted Digit", predicted)
        st.metric("Confidence", f"{confidence:.1f}%")