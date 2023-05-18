import numpy as np
import torch
from torch.utils.data import Dataset
from os import listdir
from PIL import Image
import random
from glob import glob 
import os
# import staintools
import math


class DataProcessor(Dataset):
    def __init__(self, imgs_dir, transformations=None):
        self.img_pairs = imgs_dir
        self.transformations = transformations
        random.shuffle(self.img_pairs) # shuflle the list

    def __getitem__(self, i):
        img_pair_dir = self.img_pairs[i]
        for i in range(16):
            if 'p' + str(i+1) + '_' in img_pair_dir: 
                label = torch.tensor(i)        
        try:
            img1_path = glob(os.path.join(img_pair_dir, '*-10x.png'))[0]
            img2_path = glob(os.path.join(img_pair_dir, '*-2.5x.png'))[0]
        except:
            img1_path = glob(os.path.join(img_pair_dir, '*-20x.png'))[0]
            img2_path = glob(os.path.join(img_pair_dir, '*-5x.png'))[0]
        
        img1_to_transform = np.asarray(Image.open(img1_path))
        img2_to_transform = np.asarray(Image.open(img2_path))
       
        img1_array = self.transformations(img1_to_transform)
        img2_array = self.transformations(img2_to_transform)

        combined_img = np.concatenate((img1_array, img2_array), axis = 0)
        return {"image": combined_img, "label": label}


    def __len__(self):
        return len(self.img_pairs)

