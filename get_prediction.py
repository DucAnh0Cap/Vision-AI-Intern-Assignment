import os
import torch
from torchvision import transforms
from model import Simple_CNN
from config.utils import get_config
from PIL import Image

config = get_config('config/basic_cnn.yaml')


def load_checkpoint(checkpoint_path: str):
    checkpoint = None  # Initialize checkpoint to None
    if os.path.isfile(os.path.join(checkpoint_path, "last_model.pth")):
        checkpoint = os.path.join(checkpoint_path, "last_model.pth")
    if checkpoint is None: # Check if checkpoint was assigned
        raise FileNotFoundError(f"Checkpoint file not found in {checkpoint_path}")
    checkpoint = torch.load(checkpoint,
                            weights_only=False,
                            map_location=torch.device("cpu"))
    return checkpoint


def predict(image_path,
            config=config,
            checkpoint_path='saved_models'):
    checkpoint = load_checkpoint(checkpoint_path)
    model = Simple_CNN(config.MODEL)
    model.load_state_dict(checkpoint['state_dict'], # Load the state_dict from the checkpoint file
                          strict=False)

    image = Image.open(image_path)

    transform = transforms.Compose([
            transforms.Resize(256),       # keep aspect ratio
            transforms.CenterCrop(224),   # crop center
            transforms.ToTensor(),
        ])

    input = transform(image).unsqueeze(0)

    model.eval()
    with torch.inference_mode():
        output = model(input)

    output = output.argmax(dim=-1)
    print(output)