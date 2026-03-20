import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import numpy as np
from PIL import Image
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
    .stApp { background: #0a0a0f; }
    h1, h2, h3 { font-family: 'Syne', sans-serif !important; font-weight: 800 !important; }

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
    hr { border-color: #222240 !important; }
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
# CNN Model (PyTorch)
# =========================
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# =========================
# Train model (cached)
# =========================
@st.cache_resource
def load_and_train_model():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Download MNIST
    train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_data  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    # Load custom dataset if exists
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(BASE_DIR, "dataset")
    custom_loaded = False

    all_train = train_data

    if os.path.exists(dataset_path) and os.listdir(dataset_path):
        custom_images, custom_labels = [], []
        for label in os.listdir(dataset_path):
            folder = os.path.join(dataset_path, label)
            if not os.path.isdir(folder):
                continue
            for file in os.listdir(folder):
                try:
                    img = Image.open(os.path.join(folder, file)).convert("L").resize((28, 28))
                    arr = np.array(img, dtype=np.float32) / 255.0
                    if np.mean(arr) > 0.5:
                        arr = 1 - arr
                    arr = (arr > 0.1).astype(np.float32)
                    # Normalize same as MNIST
                    arr = (arr - 0.1307) / 0.3081
                    custom_images.append(arr)
                    custom_labels.append(int(label))
                except:
                    pass

        if custom_images:
            imgs_tensor   = torch.tensor(np.array(custom_images)).unsqueeze(1)
            labels_tensor = torch.tensor(custom_labels, dtype=torch.long)
            custom_dataset = TensorDataset(imgs_tensor, labels_tensor)
            all_train = ConcatDataset([train_data, custom_dataset])
            custom_loaded = True

    train_loader = DataLoader(all_train, batch_size=64, shuffle=True)
    test_loader  = DataLoader(test_data,  batch_size=64, shuffle=False)

    # Train
    model     = CNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(5):
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs   = model(images)
            _, predicted = torch.max(outputs, 1)
            total    += labels.size(0)
            correct  += (predicted == labels).sum().item()

    accuracy = correct / total

    # Save test images for demo
    test_images = test_data.data.numpy() / 255.0
    test_labels = test_data.targets.numpy()

    return model, accuracy, custom_loaded, test_images, test_labels


# =========================
# Predict function
# =========================
def predict_digit(model, img: Image.Image):
    img = img.convert("L").resize((28, 28))
    arr = np.array(img, dtype=np.float32) / 255.0

    if np.mean(arr) > 0.5:
        arr = 1 - arr

    arr = (arr > 0.1).astype(np.float32)
    arr = (arr - 0.1307) / 0.3081

    tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(tensor)
        probs  = torch.softmax(output, dim=1).numpy()[0]

    top3      = np.argsort(probs)[::-1][:3]
    predicted = int(np.argmax(probs))
    confidence = float(probs[predicted]) * 100

    return predicted, confidence, [(int(i), float(probs[i]) * 100) for i in top3]


# =========================
# UI
# =========================
st.markdown('<p class="big-title">DIGIT PREDICTOR</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI in Robotics — CNN trained on MNIST</p>', unsafe_allow_html=True)
st.markdown("---")

with st.spinner("🧠 Training model on MNIST... (first load only, ~1-2 minutes)"):
    model, accuracy, custom_loaded, test_images, test_labels = load_and_train_model()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Model Accuracy", f"{accuracy * 100:.1f}%")
with col2:
    st.metric("Training Images", "60,000+")
with col3:
    st.metric("Custom Dataset", "✅ Loaded" if custom_loaded else "❌ Not found")

st.markdown("---")

# =========================
# Mode selection
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

        BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(BASE_DIR, "dataset")

        if not os.path.exists(dataset_path):
            st.error("Dataset folder not found.")
        else:
            digit_choice = st.selectbox("Which digit?", list(range(10)))
            digit_folder = os.path.join(dataset_path, str(digit_choice))

            if os.path.exists(digit_folder):
                files = [f for f in os.listdir(digit_folder)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if files:
                    img_choice    = st.selectbox("Which image?", files)
                    img_path      = os.path.join(digit_folder, img_choice)
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
            This model was trained on <b>MNIST-style digits</b> — white strokes on black background.
            Your photo has a gray paper background and colored ink which is very different.
            The prediction might be wrong!
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")
    predict_btn = st.button("🔍 Predict", disabled=(img_to_predict is None))

with right:
    st.markdown('<p class="section-label">Prediction result</p>', unsafe_allow_html=True)

    if predict_btn and img_to_predict is not None:
        with st.spinner("Predicting..."):
            predicted, confidence, top3 = predict_digit(model, img_to_predict)

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
                <span style="font-family: Space Mono; font-size:1.1rem; width:20px;
                      color:{'#00f5a0' if digit == predicted else '#aaa'}">{digit}</span>
                <div class="confidence-bar-container" style="flex:1;">
                    <div style="background:{bar_color}; width:{min(conf, 100):.1f}%; height:100%;
                         border-radius:8px;"></div>
                </div>
                <span style="font-family: Space Mono; font-size:0.85rem; color:#666;
                      width:55px">{conf:.1f}%</span>
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
# Explanation
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
        Trained on <b>60,000 handwritten digit images</b>, all white digit
        on black background, 28×28 pixels, centered.
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
    To make it work on real phone photos, you'd need thousands of real phone photos too.
    This is called <b>domain mismatch</b> — one of the most common problems in real-world AI.
</div>
""", unsafe_allow_html=True)
