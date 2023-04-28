import numpy as np
import torch
from torch.utils.data import Dataset
from os import listdir
from PIL import Image
import random
from glob import glob
import os
import cv2

def normalize(img):
    p = np.percentile(img, 90)
    normed_img = np.clip(img * 255.0 / p, 0, 255).astype(np.uint8)
    return normed_img

class DataProcessor(Dataset):
    def __init__(self, imgs_dir, transformations = None):
        self.he_imgs_ids = glob(os.path.join(imgs_dir, 'he_stain', '*.png'))
        self.h_imgs_ids = glob(os.path.join(imgs_dir, 'h_stain', '*.png'))
        self.e_imgs_ids = glob(os.path.join(imgs_dir, 'e_stain', '*.png'))

        self.transformations = transformations

    def __getitem__(self, i):
        he_img_file = self.he_imgs_ids[i]
        h_img_file = self.h_imgs_ids[i]
        e_img_file = self.e_imgs_ids[i]

        img_color = cv2.cvtColor(cv2.imread(he_img_file), cv2.COLOR_BGR2RGB) 
        img_color = normalize(img_color)

        h_img = cv2.imread(h_img_file)
        h_img = normalize(h_img)
       
        e_img = cv2.imread(e_img_file)
        e_img = normalize(e_img)
        
        img_color = self.transformations(img_color)
        h_img = self.transformations(h_img)
        e_img = self.transformations(e_img)


        return {'he_color':img_color, 'h_stain_image': h_img, 'e_stain_image': e_img}

    def __len__(self):
        return len(self.he_imgs_ids)
    



