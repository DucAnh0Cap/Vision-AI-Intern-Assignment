import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Dict, List
from PIL import Image
import json
import os


class DogCatDataset(Dataset):
    def __init__(self, json_path, config, transform=None):
        super(DogCatDataset, self).__init__()
        with open(json_path, encoding='utf-8') as f:
            json_data = json.load(f)
        
        self.annotations = self.load_annotations(json_data)

        self.folder_path = config.IMAGE_FOLDER
        self.transform = transform

    def load_annotations(self, json_data: Dict) -> List:
        annotations = list()

        for k, v in json_data.items():
            annotation = {
                'id': k,
                'image_id': v['image_name'],
                'label': v['label']
            }

            annotations.append(annotation)

        return annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        item = self.annotations[index]

        image_path = os.path.join(self.folder_path, item['image_id'] + '.jpg')

        # Check if file exixt
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Path does not exist: {image_path}")

        image = Image.open(image_path)

        # Transform image
        if self.transform:
            image = self.transform(image)
        else:
            transform = transforms.Compose([
                transforms.Resize(256),       # keep aspect ratio
                transforms.CenterCrop(224),   # crop center
                transforms.ToTensor(),
            ])
            image = transform(image)

        return {
            'image': image,
            'label': item['label'],
            'image_id': item['image_id']
        }

    def collate_fn(self, batch):
        images = torch.stack([item["image"] for item in batch])
        labels = torch.tensor([item["label"] for item in batch])
        paths = [item["path"] for item in batch]  # keep as list
        return {"image": images, "label": labels, "path": paths}
