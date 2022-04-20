import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import random
from glob import glob
import os
import cv2 as cv

class DataProcessor(Dataset):
    def __init__(self, imgs_dir, transformations=None):
        self.imgs_ids = glob(os.path.join(imgs_dir, '*', '*resized.png'))
        self.transformations = transformations
        random.shuffle(self.imgs_ids) # shuflle the list

    def __getitem__(self, i):
        img_file = self.imgs_ids[i]
        if 'lepidic' in img_file:
            label = torch.tensor(0)
        elif 'acinar' in img_file:
            label = torch.tensor(1)
        elif 'papillary' in img_file:
            label = torch.tensor(2)
        elif 'solid' in img_file:
            label = torch.tensor(3)
        elif 'nontumor' in img_file:
            label = torch.tensor(4)
        
        img = np.asarray(Image.open(img_file))
        if self.transformations:
            img_normed = self.transformations(img)

        return {"image": img_normed, "label": label}

    def __len__(self):
        return len(self.imgs_ids)
