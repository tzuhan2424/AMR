import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
from torchvision import transforms

# Dataset class
class BC(data.Dataset):
    def __init__(self, image_path, label, transform):
        self.image_path = image_path
        self.label = label
        self.transform = transform

    def __getitem__(self, index):
        img = self.transform(Image.open(self.image_path[index]).convert("RGB"))
        label = torch.from_numpy(np.array(self.label))[index]        
        return {'img': img, 'label': label}

    def __len__(self):
        return len(self.image_path)