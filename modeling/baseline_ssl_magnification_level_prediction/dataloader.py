import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import random
import os

class DataProcessor(Dataset):
    def __init__(self, imgs, transformations=None):
        self.imgs = imgs
        self.transformations = transformations
        random.shuffle(self.imgs) # shuflle the list

    def __getitem__(self, i):
        img_path = self.imgs[i]
        if 'level0' in os.path.basename(img_path):
            label = torch.tensor(0)
        elif 'level1' in os.path.basename(img_path):
            label = torch.tensor(1)
        if 'level2' in os.path.basename(img_path):
            label = torch.tensor(2)
        if 'level3' in os.path.basename(img_path):
            label = torch.tensor(3)
                        
        img_to_transform = np.asarray(Image.open(img_path))
        
        if self.transformations:
            img_array = self.transformations(img_to_transform)
            
        return {"image": img_array, "label": label}

    def __len__(self):
        return len(self.imgs)

