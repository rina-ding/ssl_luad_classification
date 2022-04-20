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
import math

class DataProcessor(Dataset):
    def __init__(self, imgs_dir, transformations=None):
        self.imgs = glob(os.path.join(imgs_dir, '*', '*.png'))
        self.transformations = transformations
        random.shuffle(self.imgs) # shuflle the list

    def __getitem__(self, i):
        img_path = self.imgs[i]
        img_array = np.asarray(Image.open(img_path))
        img1_array_aug = self.transformations(img_array)
        img2_array_aug = self.transformations(img_array)
        

        return {"image1": img1_array_aug, "image2": img2_array_aug}

    def __len__(self):
        return len(self.imgs)

