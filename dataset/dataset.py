import torch
from torch.utils.data import Dataset
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
                'image_id': v['image_id'],
                'label': v['label']
            }
        
            annotations.append(annotation)
        
        return annotations
    
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        item = self.annotations[index]

        image_path = os.path.join(self.folder_path, item['image_id'] + '.jpg')
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Path does not exist: {image_path}")
        
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        label = item['label']

        return image, label
