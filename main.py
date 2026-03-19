import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

st.title("🔢 MNIST Digit Recognizer")
st.write("Upload a handwritten digit image and the model will predict it!")

# ─── Define CNN Model ───────────────────────────────────────────────────────
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.fc(self.conv(x))

# ─── Load & Train (cached) ──────────────────────────────────────────────────
@st.cache_resource
def load_model():
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
    test_data  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    model = CNN()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(3):
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

    model.eval()

    test_images = test_data.data.numpy() / 255.0
    test_labels = test_data.targets.numpy()

    return model, test_images, test_labels

with st.spinner("Training model on MNIST... (first load only, ~1-2 min)"):
    model, test_images, test_labels = load_model()

st.success("Model ready!")

# ─── Section 1: Test with dataset image ─────────────────────────────────────
st.header("1. Test with a Dataset Image")
index = st.slider("Pick an image index from test set", 0, len(test_images) - 1, 0)

image = test_images[index]
true_label = test_labels[index]

tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
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

# ─── Section 2: Upload your own image ───────────────────────────────────────
st.header("2. Upload Your Own Digit Image")
uploaded = st.file_uploader("Upload a handwritten digit (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("L").resize((28, 28))
    img_array = np.array(img) / 255.0

    if np.mean(img_array) > 0.5:
        img_array = 1 - img_array

    img_array = (img_array > 0.3).astype(float)

    tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1).numpy()[0]
        predicted = int(np.argmax(probs))
        confidence = float(np.max(probs)) * 100

    col1, col2 = st.columns(2)
    with col1:
        st.image(img_array, width=200, caption="Your Image", clamp=True)
    with col2:
        st.metric("Predicted Digit", predicted)
        st.metric("Confidence", f"{confidence:.1f}%")