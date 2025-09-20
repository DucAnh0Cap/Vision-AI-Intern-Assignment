import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from model import Simple_CNN
from config.utils import get_config
import os

# Load config
config = get_config("config/basic_cnn.yaml")

# Load model
checkpoint_path = os.path.join(config.TRAINING.CHECKPOINT_PATH, "best_model.pth")
checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

model = Simple_CNN(config.MODEL)
model.load_state_dict(checkpoint["state_dict"], strict=False)
model.eval()

# Load labels
labels = getattr(config, "LABELS", ["Cat", "Dog"])

# Image transform
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

st.title("Dog vs Cat Classifier")
st.write("Upload an image and the model will predict whether it's a Dog or a Cat.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0)

    with torch.inference_mode():
        output = model(input_tensor)
        pred = output.argmax(dim=-1).item()

    st.markdown(f"### Prediction: **{labels[pred]}** 🐾")
