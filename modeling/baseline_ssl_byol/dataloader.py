import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import random

class DataProcessor(Dataset):
    def __init__(self, imgs_dir, transformations=None):
        self.imgs = imgs_dir
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

