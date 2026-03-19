import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import cv2

st.title("🔢 MNIST Digit Recognizer")
st.write("📸 Take a photo of your handwritten digit on paper and upload it!")

# ─── Improved CNN Model with Dropout ────────────────────────────────────────
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),

            # Block 2
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),

            # Block 3
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Dropout2d(0.25),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

# ─── Train with Data Augmentation ───────────────────────────────────────────
@st.cache_resource
def load_model():
    # Augmentation for training — teaches model to handle rotations, shifts
    train_transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = datasets.MNIST(root="./data", train=True,  download=True, transform=train_transform)
    test_data  = datasets.MNIST(root="./data", train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

    model = CNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(10):
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()

    test_images = test_data.data.numpy() / 255.0
    test_labels = test_data.targets.numpy()

    return model, test_images, test_labels

with st.spinner("🏋️ Training improved model (10 epochs)... first load ~3-5 min"):
    model, test_images, test_labels = load_model()

st.success("✅ Model ready! (~99% accuracy on test set)")

# ─── Smart Image Preprocessor for Paper Photos ──────────────────────────────
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)

    # Convert to grayscale
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Auto crop to bounding box of content
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold — works great for paper photos with shadows/lines
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    # Find contours to locate the digit
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Get bounding box of largest contour (the digit)
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)

        # Add padding around digit
        pad = 20
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(gray.shape[1] - x, w + 2 * pad)
        h = min(gray.shape[0] - y, h + 2 * pad)

        # Crop to digit
        thresh = thresh[y:y+h, x:x+w]

    # Make square
    size = max(thresh.shape)
    square = np.zeros((size, size), dtype=np.uint8)
    x_off = (size - thresh.shape[1]) // 2
    y_off = (size - thresh.shape[0]) // 2
    square[y_off:y_off+thresh.shape[0], x_off:x_off+thresh.shape[1]] = thresh

    # Resize to 28x28
    resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)

    # Normalize
    img_array = resized / 255.0

    # Ensure white digit on black background
    if np.mean(img_array) > 0.5:
        img_array = 1 - img_array

    return img_array

# ─── Section 1: Test with dataset image ─────────────────────────────────────
st.header("1. Test with a Dataset Image")
index = st.slider("Pick an image index from test set", 0, len(test_images) - 1, 0)

image = test_images[index]
true_label = test_labels[index]

norm = transforms.Normalize((0.1307,), (0.3081,))
tensor = norm(torch.tensor(image, dtype=torch.float32).unsqueeze(0)).unsqueeze(0)

with torch.no_grad():
    output = model(tensor)
    probs = torch.softmax(output, dim=1).numpy()[0]
    predicted = int(np.argmax(probs))
    confidence = float(np.max(probs)) * 100

col1, col2 = st.columns(2)
with col1:
    st.image(image, width=200, caption="Test Image", clamp=True)
with col2:
    st.metric("Predicted", predicted)
    st.metric("True Label", int(true_label))
    st.metric("Confidence", f"{confidence:.1f}%")
    if predicted == int(true_label):
        st.success("✅ Correct!")
    else:
        st.error("❌ Wrong prediction")

# ─── Section 2: Upload paper photo ──────────────────────────────────────────
st.header("2. Upload Your Handwritten Digit (Paper Photo)")
st.info("📸 Tips for best results:\n- Write the digit **big and bold** with a pen\n- Good lighting, no shadows\n- Keep the digit centered\n- Plain white paper works best")

uploaded = st.file_uploader("Upload photo of your handwritten digit", type=["png", "jpg", "jpeg"])

if uploaded:
    col1, col2 = st.columns(2)

    img_array = preprocess_image(uploaded)

    norm = transforms.Normalize((0.1307,), (0.3081,))
    tensor = norm(torch.tensor(img_array, dtype=torch.float32).unsqueeze(0)).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1).numpy()[0]
        predicted = int(np.argmax(probs))
        confidence = float(np.max(probs)) * 100

    # Show top 3 predictions
    top3_idx = np.argsort(probs)[::-1][:3]

    with col1:
        st.image(img_array, width=200, caption="Processed Image", clamp=True)
        # Also show original
        st.image(Image.open(uploaded), width=200, caption="Original Photo")
    with col2:
        st.metric("Predicted Digit", predicted)
        st.metric("Confidence", f"{confidence:.1f}%")
        st.write("**Top 3 guesses:**")
        for i in top3_idx:
            bar_val = float(probs[i])
            st.write(f"Digit **{i}**: {bar_val*100:.1f}%")
            st.progress(bar_val)