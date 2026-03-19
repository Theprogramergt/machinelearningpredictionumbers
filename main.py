import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2

st.title("🔢 MNIST Digit Recognizer")
st.write("🧠 Deep Neural Network — trained on 280,000 handwriting samples!")

# ─── Deep CNN Model ──────────────────────────────────────────────────────────
class DeepCNN(nn.Module):
    def __init__(self):
        super(DeepCNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),

            # Block 2
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),

            # Block 3
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# ─── Train on MNIST + EMNIST combined ───────────────────────────────────────
@st.cache_resource
def load_model():
    # Heavy augmentation — handles rotations, different pen pressures, styles
    train_transform = transforms.Compose([
        transforms.RandomRotation(20),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.15, 0.15),
            scale=(0.85, 1.15),
            shear=10
        ),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # MNIST dataset (60,000 samples)
    mnist_train = datasets.MNIST(
        root="./data", train=True, download=True, transform=train_transform
    )

    # EMNIST digits dataset (280,000 samples — more diverse handwriting styles)
    emnist_train = datasets.EMNIST(
        root="./data", split="digits", train=True, download=True, transform=train_transform
    )

    # Combine both datasets = 340,000 total training samples
    combined = ConcatDataset([mnist_train, emnist_train])
    train_loader = DataLoader(combined, batch_size=256, shuffle=True, num_workers=0)

    # Test set
    mnist_test = datasets.MNIST(
        root="./data", train=False, download=True, transform=test_transform
    )
    test_loader = DataLoader(mnist_test, batch_size=256, shuffle=False)

    model = DeepCNN()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.01,
        steps_per_epoch=len(train_loader),
        epochs=8
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Training loop
    progress = st.progress(0)
    status = st.empty()

    model.train()
    total_steps = 8 * len(train_loader)
    step = 0

    for epoch in range(8):
        correct = 0
        total = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            step += 1
            progress.progress(step / total_steps)

        acc = 100. * correct / total
        status.text(f"Epoch {epoch+1}/8 — Training accuracy: {acc:.1f}%")

    progress.empty()
    status.empty()

    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_acc = 100. * correct / total

    test_images = mnist_test.data.numpy() / 255.0
    test_labels = mnist_test.targets.numpy()

    return model, test_images, test_labels, test_acc

with st.spinner("🧠 Training Deep Neural Network on 340,000 samples... (~5-8 min first load)"):
    model, test_images, test_labels, test_acc = load_model()

st.success(f"✅ Neural Network ready! Test accuracy: {test_acc:.2f}%")

# ─── Smart preprocessor for paper photos ─────────────────────────────────────
def preprocess_paper_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    # Adaptive threshold — handles shadows, grid lines, uneven lighting
    blur = cv2.GaussianBlur(denoised, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15, C=4
    )

    # Remove grid lines using morphological operations
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    vertical_kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel,   iterations=2)
    grid    = cv2.add(h_lines, v_lines)
    clean   = cv2.subtract(thresh, grid)

    # Find digit contour
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        pad = int(max(w, h) * 0.25)
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(clean.shape[1] - x, w + 2 * pad)
        h = min(clean.shape[0] - y, h + 2 * pad)
        clean = clean[y:y+h, x:x+w]

    # Make square with padding
    size = max(clean.shape[0], clean.shape[1])
    square = np.zeros((size, size), dtype=np.uint8)
    x_off = (size - clean.shape[1]) // 2
    y_off = (size - clean.shape[0]) // 2
    square[y_off:y_off+clean.shape[0], x_off:x_off+clean.shape[1]] = clean

    # Resize to 28x28
    resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
    img_array = resized / 255.0

    # Ensure black background white digit
    if np.mean(img_array) > 0.5:
        img_array = 1 - img_array

    return img_array

# ─── Normalize helper ─────────────────────────────────────────────────────────
norm = transforms.Normalize((0.1307,), (0.3081,))

def predict(img_array):
    t = norm(torch.tensor(img_array, dtype=torch.float32).unsqueeze(0)).unsqueeze(0)
    with torch.no_grad():
        output = model(t)
        probs  = torch.softmax(output, dim=1).numpy()[0]
    return probs

# ─── Section 1: Dataset test ──────────────────────────────────────────────────
st.header("1. Test with Dataset Image")
index = st.slider("Pick test image", 0, len(test_images) - 1, 0)

image      = test_images[index]
true_label = int(test_labels[index])
probs      = predict(image)
predicted  = int(np.argmax(probs))
confidence = float(np.max(probs)) * 100

col1, col2 = st.columns(2)
with col1:
    st.image(image, width=200, caption="Test image", clamp=True)
with col2:
    st.metric("Predicted", predicted)
    st.metric("True Label", true_label)
    st.metric("Confidence", f"{confidence:.1f}%")
    st.success("✅ Correct!") if predicted == true_label else st.error("❌ Wrong")

# ─── Section 2: Upload your handwriting ──────────────────────────────────────
st.header("2. Upload Your Handwritten Digit")
st.info("📸 Works with European style, grid paper, any pen!\n\n**Tips:** One digit per photo • Good lighting • Crop close to digit")

uploaded = st.file_uploader("Upload digit photo", type=["png", "jpg", "jpeg"])

if uploaded:
    img_array = preprocess_paper_image(uploaded)
    probs     = predict(img_array)
    predicted = int(np.argmax(probs))
    confidence = float(np.max(probs)) * 100
    top3      = np.argsort(probs)[::-1][:3]

    col1, col2 = st.columns(2)
    with col1:
        st.image(Image.open(uploaded), caption="Your photo", use_column_width=True)
        st.image(img_array, width=150, caption="Processed (what AI sees)", clamp=True)
    with col2:
        st.metric("Predicted Digit", predicted)
        st.metric("Confidence", f"{confidence:.1f}%")
        st.write("**Top 3 guesses:**")
        for i in top3:
            st.write(f"Digit **{i}**: {probs[i]*100:.1f}%")
            st.progress(float(probs[i]))