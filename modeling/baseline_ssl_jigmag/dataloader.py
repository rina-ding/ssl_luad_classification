import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import random
from glob import glob 
import os

class DataProcessor(Dataset):
    def __init__(self, imgs_dir, transformations=None):
        self.img_pairs = imgs_dir
        self.transformations = transformations
        random.shuffle(self.img_pairs) # shuflle the list

    def __getitem__(self, i):
        img_pair_dir = self.img_pairs[i]
        for i in range(24):
            if 'c' + str(i+1) in os.path.basename(img_pair_dir).split('_')[1]: 
                label = torch.tensor(i)
                
        img1_path = glob(os.path.join(img_pair_dir, '*_order0.png'))[0]
        img2_path = glob(os.path.join(img_pair_dir, '*_order1.png'))[0]
        img3_path = glob(os.path.join(img_pair_dir, '*_order2.png'))[0]
        img4_path = glob(os.path.join(img_pair_dir, '*_order3.png'))[0]

        img1_to_transform = np.asarray(Image.open(img1_path))
        img2_to_transform = np.asarray(Image.open(img2_path))
        img3_to_transform = np.asarray(Image.open(img3_path))
        img4_to_transform = np.asarray(Image.open(img4_path))
        
        if self.transformations:
            img1_array = self.transformations(img1_to_transform)
            img2_array = self.transformations(img2_to_transform)
            img3_array = self.transformations(img3_to_transform)
            img4_array = self.transformations(img4_to_transform)

        combined_img = np.concatenate((img1_array, img2_array, img3_array, img4_array), axis = 0)
        return {"image": combined_img, "label": label}

    def __len__(self):
        return len(self.img_pairs)

