import os
import argparse
import torch
from torchvision import transforms
from model import Simple_CNN
from config.utils import get_config
from PIL import Image


def predict(image_path, config, checkpoint_path):
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

    # Transform
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    # Labels
    labels = getattr(config, "LABELS", ["Cat", "Dog"])

    def _predict_single(img_path):
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        
        image = Image.open(img_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        model.eval()
        with torch.inference_mode():
            output = model(input_tensor)

        pred = output.argmax(dim=-1).item()
        return labels[pred]

    # If folder → loop through images
    if os.path.isdir(image_path):
        results = {}
        for f_ in os.listdir(image_path):
            fpath = os.path.join(image_path, f_)
            if fpath.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                try:
                    results[f_] = _predict_single(fpath)
                except Exception as e:
                    results[f_] = f"Error: {e}"
        return results
    else:
        # Single image
        return _predict_single(image_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, default="config/basic_cnn.yaml")
    parser.add_argument("--image-file", type=str, required=True,
                        help="Path to an image file or a folder of images")
    parser.add_argument("--checkpoint", type=str, default="saved_models/best_model.pth")
    args = parser.parse_args()

    config = get_config(args.config_file)
    predictions = predict(args.image_file, config, args.checkpoint)

    if isinstance(predictions, dict):  # folder case
        for fname, pred in predictions.items():
            print(f"{fname}: {pred}")
    else:
        print(predictions)
