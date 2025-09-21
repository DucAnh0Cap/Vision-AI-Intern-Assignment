import os
import argparse
import torch
from torchvision import transforms
from model import Simple_CNN
from config.utils import get_config
from PIL import Image


def predict(image_path,
            config,
            checkpoint_path):

    # Load checkpoint
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Load model
    model = Simple_CNN(config.MODEL)
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load and transform image
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Inference
    model.eval()
    with torch.inference_mode():
        output = model(input_tensor)

    pred = output.argmax(dim=-1).item()
    labels = getattr(config, "LABELS", ["Cat", "Dog"])
    return labels[pred]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, default="config/basic_cnn.yaml")
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="saved_models/best_model.pth")
    args = parser.parse_args()

    config = get_config(args.config_file)
    prediction = predict(args.image_file, config, args.checkpoint)
    print(prediction)