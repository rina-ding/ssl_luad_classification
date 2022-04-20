import numpy as np
import torch
from torch.utils.data import Dataset
from os import listdir
from skimage import io
from PIL import Image
import random
from glob import glob
import os
import math


class DataProcessor(Dataset):
    def __init__(self, imgs_dir, transformations=None):
        self.img_pairs = glob(os.path.join(imgs_dir, '*_20_10_*')) # We are only using 20x-10x pairs for now
        self.transformations = transformations
        random.shuffle(self.img_pairs) # shuflle the list

    def __getitem__(self, i):
        img_pair_dir = self.img_pairs[i]
        # The class where one tile is NOT part of the other tile
        if "no" in img_pair_dir: 
            label = torch.tensor([0])
            # Make sure that the 20x tile always comes first so that the ordering of the 2 tiles is consistent in training
            img1_path = glob(os.path.join(img_pair_dir, '*20x.png'))[0]
            img2_path = glob(os.path.join(img_pair_dir, '*10x.png'))[0]

        # The class where one tile is part of the other tile
        else:
            # Make sure that the 20x tile always comes first so that the ordering of the 2 tiles is consistent in training
            label = torch.tensor([1])
            img1_path = glob(os.path.join(img_pair_dir, '*20x.png'))[0]
            img2_path = glob(os.path.join(img_pair_dir, '*10x.png'))[0] 
            
        img1_to_transform = np.asarray(Image.open(img1_path))
        img2_to_transform = np.asarray(Image.open(img2_path))
        if self.transformations:
            img1_array = self.transformations(img1_to_transform)
            img2_array = self.transformations(img2_to_transform)

        combined_img = np.concatenate((img1_array, img2_array), axis = 0)
        return {"image": combined_img, "label": label}

    def __len__(self):
        return len(self.img_pairs)

