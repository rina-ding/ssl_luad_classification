import numpy as np
import torch
from torch.utils.data import Dataset
from os import listdir
from PIL import Image
import random
from glob import glob
import os
import math
import torchvision.transforms as transforms


class DataProcessor(Dataset):
    def __init__(self, imgs_dir, transformations=None):
        self.img_pairs = imgs_dir
        self.transformations = transformations
        random.shuffle(self.img_pairs) # shuflle the list

    def __getitem__(self, i):
        img_pair_dir = self.img_pairs[i]
        all_imgs = glob(os.path.join(img_pair_dir, '*.png'))
        random.shuffle(all_imgs)

        if 'no' in img_pair_dir and 'higher_mag_tile.png' in os.path.basename(all_imgs[0]):
            label = torch.tensor(0)
        elif 'yes' in img_pair_dir and 'higher_mag_tile.png' in os.path.basename(all_imgs[0]):
            label = torch.tensor(1)
        elif 'no' in img_pair_dir and 'lower_mag_tile.png' in os.path.basename(all_imgs[0]):
            label = torch.tensor(0) + torch.tensor(2)
        elif 'yes' in img_pair_dir and 'lower_mag_tile.png' in os.path.basename(all_imgs[0]):
            label = torch.tensor(1) + torch.tensor(2)
            
        img1_path = all_imgs[0]
        img2_path = all_imgs[1]
            
        img1_to_transform = np.asarray(Image.open(img1_path))
        img2_to_transform = np.asarray(Image.open(img2_path))
        
        if self.transformations:
            img1_array = self.transformations(img1_to_transform)
            img2_array = self.transformations(img2_to_transform)
        else:
            basic_transforms = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),                     
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
            img1_array = basic_transforms(img1_to_transform)
            img2_array = basic_transforms(img2_to_transform)

        combined_img = np.concatenate((img1_array, img2_array), axis = 0)
        return {"image": combined_img, "label": label}

    def __len__(self):
        return len(self.img_pairs)

