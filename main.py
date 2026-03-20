# import streamlit as st
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader, ConcatDataset
# import numpy as np
# from PIL import Image, ImageEnhance, ImageFilter
# import cv2
#
# st.title("🔢 MNIST Digit Recognizer")
# st.write("🧠 Deep Neural Network — trained on 280,000 handwriting samples!")
#
# # ─── Deep CNN Model ──────────────────────────────────────────────────────────
# class DeepCNN(nn.Module):
#     def __init__(self):
#         super(DeepCNN, self).__init__()
#         self.features = nn.Sequential(
#             # Block 1
#             nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
#             nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
#
#             # Block 2
#             nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
#             nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
#             nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
#
#             # Block 3
#             nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
#             nn.Dropout2d(0.25),
#         )
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(256 * 7 * 7, 512), nn.ReLU(inplace=True), nn.Dropout(0.5),
#             nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(0.4),
#             nn.Linear(256, 10)
#         )
#
#     def forward(self, x):
#         return self.classifier(self.features(x))
#
# # ─── Train on MNIST + EMNIST combined ───────────────────────────────────────
# @st.cache_resource
# def load_model():
#     # Heavy augmentation — handles rotations, different pen pressures, styles
#     train_transform = transforms.Compose([
#         transforms.RandomRotation(20),
#         transforms.RandomAffine(
#             degrees=0,
#             translate=(0.15, 0.15),
#             scale=(0.85, 1.15),
#             shear=10
#         ),
#         transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
#         transforms.ToTensor(),
#         transforms.Normalize((0.1307,), (0.3081,))
#     ])
#
#     test_transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.1307,), (0.3081,))
#     ])
#
#     # MNIST dataset (60,000 samples)
#     mnist_train = datasets.MNIST(
#         root="./data", train=True, download=True, transform=train_transform
#     )
#
#     # EMNIST digits dataset (280,000 samples — more diverse handwriting styles)
#     emnist_train = datasets.EMNIST(
#         root="./data", split="digits", train=True, download=True, transform=train_transform
#     )
#
#     # Combine both datasets = 340,000 total training samples
#     combined = ConcatDataset([mnist_train, emnist_train])
#     train_loader = DataLoader(combined, batch_size=256, shuffle=True, num_workers=0)
#
#     # Test set
#     mnist_test = datasets.MNIST(
#         root="./data", train=False, download=True, transform=test_transform
#     )
#     test_loader = DataLoader(mnist_test, batch_size=256, shuffle=False)
#
#     model = DeepCNN()
#     optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
#     scheduler = optim.lr_scheduler.OneCycleLR(
#         optimizer, max_lr=0.01,
#         steps_per_epoch=len(train_loader),
#         epochs=8
#     )
#     criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
#
#     # Training loop
#     progress = st.progress(0)
#     status = st.empty()
#
#     model.train()
#     total_steps = 8 * len(train_loader)
#     step = 0
#
#     for epoch in range(8):
#         correct = 0
#         total = 0
#         for images, labels in train_loader:
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             optimizer.step()
#             scheduler.step()
#
#             _, predicted = outputs.max(1)
#             total += labels.size(0)
#             correct += predicted.eq(labels).sum().item()
#             step += 1
#             progress.progress(step / total_steps)
#
#         acc = 100. * correct / total
#         status.text(f"Epoch {epoch+1}/8 — Training accuracy: {acc:.1f}%")
#
#     progress.empty()
#     status.empty()
#
#     # Evaluate
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for images, labels in test_loader:
#             outputs = model(images)
#             _, predicted = outputs.max(1)
#             total += labels.size(0)
#             correct += predicted.eq(labels).sum().item()
#
#     test_acc = 100. * correct / total
#
#     test_images = mnist_test.data.numpy() / 255.0
#     test_labels = mnist_test.targets.numpy()
#
#     return model, test_images, test_labels, test_acc
#
# with st.spinner("🧠 Training Deep Neural Network on 340,000 samples... (~5-8 min first load)"):
#     model, test_images, test_labels, test_acc = load_model()
#
# st.success(f"✅ Neural Network ready! Test accuracy: {test_acc:.2f}%")
#
# # ─── Smart preprocessor for paper photos ─────────────────────────────────────
# def preprocess_paper_image(uploaded_file):
#     img = Image.open(uploaded_file).convert("RGB")
#     img_np = np.array(img)
#
#     gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
#
#     # Denoise
#     denoised = cv2.fastNlMeansDenoising(gray, h=10)
#
#     # Adaptive threshold — handles shadows, grid lines, uneven lighting
#     blur = cv2.GaussianBlur(denoised, (5, 5), 0)
#     thresh = cv2.adaptiveThreshold(
#         blur, 255,
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY_INV,
#         blockSize=15, C=4
#     )
#
#     # Remove grid lines using morphological operations
#     horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
#     vertical_kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
#     h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
#     v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel,   iterations=2)
#     grid    = cv2.add(h_lines, v_lines)
#     clean   = cv2.subtract(thresh, grid)
#
#     # Find digit contour
#     contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     if contours:
#         largest = max(contours, key=cv2.contourArea)
#         x, y, w, h = cv2.boundingRect(largest)
#         pad = int(max(w, h) * 0.25)
#         x = max(0, x - pad)
#         y = max(0, y - pad)
#         w = min(clean.shape[1] - x, w + 2 * pad)
#         h = min(clean.shape[0] - y, h + 2 * pad)
#         clean = clean[y:y+h, x:x+w]
#
#     # Make square with padding
#     size = max(clean.shape[0], clean.shape[1])
#     square = np.zeros((size, size), dtype=np.uint8)
#     x_off = (size - clean.shape[1]) // 2
#     y_off = (size - clean.shape[0]) // 2
#     square[y_off:y_off+clean.shape[0], x_off:x_off+clean.shape[1]] = clean
#
#     # Resize to 28x28
#     resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
#     img_array = resized / 255.0
#
#     # Ensure black background white digit
#     if np.mean(img_array) > 0.5:
#         img_array = 1 - img_array
#
#     return img_array
#
# # ─── Normalize helper ─────────────────────────────────────────────────────────
# norm = transforms.Normalize((0.1307,), (0.3081,))
#
# def predict(img_array):
#     t = norm(torch.tensor(img_array, dtype=torch.float32).unsqueeze(0)).unsqueeze(0)
#     with torch.no_grad():
#         output = model(t)
#         probs  = torch.softmax(output, dim=1).numpy()[0]
#     return probs
#
# # ─── Section 1: Dataset test ──────────────────────────────────────────────────
# st.header("1. Test with Dataset Image")
# index = st.slider("Pick test image", 0, len(test_images) - 1, 0)
#
# image      = test_images[index]
# true_label = int(test_labels[index])
# probs      = predict(image)
# predicted  = int(np.argmax(probs))
# confidence = float(np.max(probs)) * 100
#
# col1, col2 = st.columns(2)
# with col1:
#     st.image(image, width=200, caption="Test image", clamp=True)
# with col2:
#     st.metric("Predicted", predicted)
#     st.metric("True Label", true_label)
#     st.metric("Confidence", f"{confidence:.1f}%")
#     st.success("✅ Correct!") if predicted == true_label else st.error("❌ Wrong")
#
# # ─── Section 2: Upload your handwriting ──────────────────────────────────────
# st.header("2. Upload Your Handwritten Digit")
# st.info("📸 Works with European style, grid paper, any pen!\n\n**Tips:** One digit per photo • Good lighting • Crop close to digit")
#
# uploaded = st.file_uploader("Upload digit photo", type=["png", "jpg", "jpeg"])
#
# if uploaded:
#     img_array = preprocess_paper_image(uploaded)
#     probs     = predict(img_array)
#     predicted = int(np.argmax(probs))
#     confidence = float(np.max(probs)) * 100
#     top3      = np.argsort(probs)[::-1][:3]
#
#     col1, col2 = st.columns(2)
#     with col1:
#         st.image(Image.open(uploaded), caption="Your photo", use_column_width=True)
#         st.image(img_array, width=150, caption="Processed (what AI sees)", clamp=True)
#     with col2:
#         st.metric("Predicted Digit", predicted)
#         st.metric("Confidence", f"{confidence:.1f}%")
#         st.write("**Top 3 guesses:**")
#         for i in top3:
#             st.write(f"Digit **{i}**: {probs[i]*100:.1f}%")
#             st.progress(float(probs[i]))





import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from PIL import Image, ImageFilter
import os

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="Digit Predictor",
    page_icon="🔢",
    layout="wide"
)

# =========================
# Custom CSS
# =========================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Syne', sans-serif;
        background-color: #0a0a0f;
        color: #f0f0f0;
    }

    .stApp {
        background: #0a0a0f;
    }

    h1, h2, h3 {
        font-family: 'Syne', sans-serif !important;
        font-weight: 800 !important;
    }

    .big-title {
        font-family: 'Syne', sans-serif;
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00f5a0, #00d9f5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }

    .subtitle {
        font-family: 'Space Mono', monospace;
        font-size: 0.85rem;
        color: #666;
        margin-top: 0;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .prediction-box {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #00f5a0;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }

    .prediction-number {
        font-family: 'Space Mono', monospace;
        font-size: 5rem;
        font-weight: 700;
        color: #00f5a0;
        line-height: 1;
    }

    .confidence-bar-container {
        background: #1a1a2e;
        border-radius: 8px;
        height: 12px;
        margin: 4px 0;
        overflow: hidden;
    }

    .info-card {
        background: #111120;
        border: 1px solid #222240;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.5rem 0;
    }

    .warning-card {
        background: #1a0a00;
        border: 1px solid #ff6b35;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.5rem 0;
        color: #ff9966;
    }

    .success-card {
        background: #001a0a;
        border: 1px solid #00f5a0;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.5rem 0;
        color: #00f5a0;
    }

    .digit-chip {
        display: inline-block;
        background: #1a1a2e;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 4px 12px;
        font-family: 'Space Mono', monospace;
        font-size: 0.8rem;
        margin: 2px;
        color: #aaa;
    }

    .stButton > button {
        background: linear-gradient(135deg, #00f5a0, #00d9f5) !important;
        color: #0a0a0f !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.5rem !important;
        font-size: 1rem !important;
    }

    .stButton > button:hover {
        opacity: 0.85 !important;
        transform: translateY(-1px) !important;
    }

    div[data-testid="stFileUploader"] {
        background: #111120;
        border: 2px dashed #333;
        border-radius: 12px;
        padding: 1rem;
    }

    .stRadio > div {
        background: #111120;
        border-radius: 8px;
        padding: 0.5rem;
    }

    hr {
        border-color: #222240 !important;
    }

    .section-label {
        font-family: 'Space Mono', monospace;
        font-size: 0.7rem;
        color: #555;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# =========================
# Model (cached so it only trains once)
# =========================
@st.cache_resource
def load_and_train_model():
    # Load MNIST
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train / 255.0
    x_test = x_test / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    # Load custom dataset if exists
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(BASE_DIR, "dataset")
    custom_loaded = False

    if os.path.exists(dataset_path) and os.listdir(dataset_path):
        images, labels = [], []
        for label in os.listdir(dataset_path):
            folder = os.path.join(dataset_path, label)
            if not os.path.isdir(folder):
                continue
            for file in os.listdir(folder):
                try:
                    img = Image.open(os.path.join(folder, file)).convert("L").resize((28, 28))
                    arr = np.array(img) / 255.0
                    if np.mean(arr) > 0.5:
                        arr = 1 - arr
                    arr = (arr > 0.1).astype(float)
                    images.append(arr)
                    labels.append(int(label))
                except:
                    pass

        if images:
            custom_x = np.array(images).reshape(-1, 28, 28, 1)
            custom_y = np.array(labels)
            x_train = np.concatenate((x_train, custom_x), axis=0)
            y_train = np.concatenate((y_train, custom_y), axis=0)
            custom_loaded = True

    # Build model
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

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5, verbose=0)
    _, acc = model.evaluate(x_test, y_test, verbose=0)

    return model, acc, custom_loaded


def preprocess_for_prediction(img: Image.Image):
    img = img.convert("L").resize((28, 28))
    arr = np.array(img) / 255.0

    if np.mean(arr) > 0.5:
        arr = 1 - arr

    arr = (arr > 0.1).astype(float)

    coords = np.argwhere(arr > 0)
    if coords.size > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        arr = arr[y_min:y_max + 1, x_min:x_max + 1]

    arr = tf.image.resize(arr[..., None], (28, 28)).numpy()
    return arr.reshape(1, 28, 28, 1)


def predict(model, img: Image.Image):
    arr = preprocess_for_prediction(img)
    preds = model.predict(arr, verbose=0)[0]
    top3 = np.argsort(preds)[::-1][:3]
    return int(np.argmax(preds)), float(np.max(preds)) * 100, [(int(i), float(preds[i]) * 100) for i in top3]


# =========================
# UI
# =========================
st.markdown('<p class="big-title">DIGIT PREDICTOR</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI in Robotics — CNN trained on MNIST</p>', unsafe_allow_html=True)
st.markdown("---")

# Train model with spinner
with st.spinner("🧠 Training model on MNIST... (first load only, ~30 seconds)"):
    model, accuracy, custom_loaded = load_and_train_model()

col_info1, col_info2, col_info3 = st.columns(3)
with col_info1:
    st.metric("Model Accuracy", f"{accuracy * 100:.1f}%")
with col_info2:
    st.metric("Training Images", "60,000+")
with col_info3:
    st.metric("Custom Dataset", "✅ Loaded" if custom_loaded else "❌ Not found")

st.markdown("---")

# =========================
# Two modes
# =========================
st.markdown('<p class="section-label">Choose test mode</p>', unsafe_allow_html=True)
mode = st.radio(
    "",
    ["📁  Use dataset image (clean, should predict correctly)",
     "📷  Upload your own image (handwritten photo)"],
    label_visibility="collapsed"
)

st.markdown("")

left, right = st.columns([1, 1], gap="large")

with left:
    img_to_predict = None

    if "dataset" in mode:
        st.markdown('<p class="section-label">Pick a digit from the dataset</p>', unsafe_allow_html=True)

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(BASE_DIR, "dataset")

        if not os.path.exists(dataset_path):
            st.error("Dataset folder not found next to this script.")
        else:
            digit_choice = st.selectbox("Which digit?", list(range(10)))
            digit_folder = os.path.join(dataset_path, str(digit_choice))

            if os.path.exists(digit_folder):
                files = [f for f in os.listdir(digit_folder)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

                if files:
                    img_choice = st.selectbox("Which image?", files)
                    img_path = os.path.join(digit_folder, img_choice)
                    img_to_predict = Image.open(img_path)
                    st.image(img_to_predict, caption=f"Selected: {img_choice}", width=200)
                else:
                    st.warning("No images found in this digit folder.")

    else:
        st.markdown('<p class="section-label">Upload your handwritten digit</p>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Upload a photo of a handwritten digit",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed"
        )
        if uploaded:
            img_to_predict = Image.open(uploaded)
            st.image(img_to_predict, caption="Your image", width=200)

            st.markdown("""
            <div class="warning-card">
            ⚠️ <b>Heads up!</b><br>
            This model was trained on <b>MNIST-style digits</b> — white strokes on black background, 
            simple and centered. Your photo has a gray paper background and blue ink, 
            which is very different. The prediction might be wrong!
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")
    predict_btn = st.button("🔍 Predict", disabled=(img_to_predict is None))

with right:
    st.markdown('<p class="section-label">Prediction result</p>', unsafe_allow_html=True)

    if predict_btn and img_to_predict is not None:
        with st.spinner("Predicting..."):
            predicted, confidence, top3 = predict(model, img_to_predict)

        st.markdown(f"""
        <div class="prediction-box">
            <div style="font-family: Space Mono; font-size:0.8rem; color:#666; letter-spacing:2px;">PREDICTED DIGIT</div>
            <div class="prediction-number">{predicted}</div>
            <div style="font-family: Space Mono; font-size:1rem; color:#aaa; margin-top:0.5rem;">
                {confidence:.1f}% confidence
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<p class="section-label" style="margin-top:1rem">Top 3 guesses</p>', unsafe_allow_html=True)
        for digit, conf in top3:
            bar_color = "#00f5a0" if digit == predicted else "#334"
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:10px; margin:6px 0;">
                <span style="font-family: Space Mono; font-size:1.1rem; width:20px; color:{'#00f5a0' if digit == predicted else '#aaa'}">{digit}</span>
                <div class="confidence-bar-container" style="flex:1;">
                    <div style="background:{bar_color}; width:{conf:.1f}%; height:100%; border-radius:8px; transition: width 0.5s ease;"></div>
                </div>
                <span style="font-family: Space Mono; font-size:0.85rem; color:#666; width:55px">{conf:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="info-card" style="text-align:center; padding: 3rem 1rem; color:#444;">
            <div style="font-size:3rem">🔢</div>
            <div style="font-family: Space Mono; font-size:0.8rem; letter-spacing:2px; margin-top:1rem;">
                SELECT AN IMAGE AND CLICK PREDICT
            </div>
        </div>
        """, unsafe_allow_html=True)

# =========================
# Explanation section
# =========================
st.markdown("---")
st.markdown("### 🧠 How it works")

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""
    <div class="info-card">
        <b>1. CNN Architecture</b><br><br>
        The model uses a <b>Convolutional Neural Network</b> with 2 Conv layers 
        that learn to detect edges, curves and shapes in the digit.
    </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown("""
    <div class="info-card">
        <b>2. MNIST Training</b><br><br>
        Trained on <b>60,000 handwritten digit images</b>, all in a specific style: 
        white digit on black background, 28×28 pixels, centered.
    </div>
    """, unsafe_allow_html=True)
with c3:
    st.markdown("""
    <div class="info-card">
        <b>3. Why photos fail</b><br><br>
        Phone photos have <b>gray backgrounds, colored ink, shadows and noise</b>. 
        The model was never trained on this style, so it struggles.
    </div>
    """, unsafe_allow_html=True)

st.markdown("")
st.markdown("""
<div class="warning-card">
    <b>📌 Key lesson:</b> A model is only as good as its training data. 
    If you train on MNIST-style images, it works perfectly on MNIST-style images. 
    To make it work on real phone photos, you'd need to train on thousands of real phone photos too.
    This is called <b>domain mismatch</b> — one of the most common problems in real-world AI.
</div>
""", unsafe_allow_html=True)