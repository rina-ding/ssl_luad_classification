from __future__ import division
import numpy as np
import torch
from torch.utils.data import Dataset
from os import listdir
# from skimage import io
from PIL import Image
import random
from glob import glob
import os
import cv2 as cv

class DataProcessor(Dataset):
    def __init__(self, imgs_dir, transformations=None):
        self.imgs = glob(os.path.join(imgs_dir, '*', '*.png'))
        self.transformations = transformations
        random.shuffle(self.imgs) # shuflle the list

    def __getitem__(self, i):
        img_path = self.imgs[i]
        img_basename = os.path.basename(img_path)
        if "5x" in img_basename: 
            label = torch.tensor(0)
        elif "10x" in img_basename: 
            label = torch.tensor(1)
        elif "20x" in img_basename: 
            label = torch.tensor(2)
        elif "40x" in img_basename: 
            label = torch.tensor(3)
        
        img_to_transform = np.asarray(Image.open(img_path))
        if self.transformations:
            img_array = self.transformations(img_to_transform)

        return {"image": img_array, "label": label}

    def __len__(self):
        return len(self.imgs)


