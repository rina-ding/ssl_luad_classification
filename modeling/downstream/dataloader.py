import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import random
import torchvision.transforms as transforms


class DataProcessor(Dataset):
    def __init__(self, imgs_dir, channel, transformations=None):
        self.imgs_ids = imgs_dir
        self.transformations = transformations
        self.channel = channel
        random.shuffle(self.imgs_ids) # shuflle the list

    def __getitem__(self, i):
        img_file = self.imgs_ids[i]
        if 'lepidic' in img_file:
            label = torch.tensor(0)
        elif 'acinar' in img_file:
            label = torch.tensor(1)
        elif 'papillary' in img_file:
            label = torch.tensor(2)
        elif 'micro' in img_file:
            label = torch.tensor(3)
        elif 'solid' in img_file:
            label = torch.tensor(4)
        elif 'nontumor' in img_file:
            label = torch.tensor(5)
        
        img = np.asarray(Image.open(img_file))

        if self.transformations:
            img_normed = self.transformations(img)
        else:      
            basic_transforms = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),   
                                                   transforms.Resize((224, 224)),                  
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
            img_normed = basic_transforms(img)

        if self.channel == 6:
            return {"image": np.concatenate((img_normed, img_normed), axis = 0), "label": label}
        elif self.channel == 12:
            return {"image": np.concatenate((img_normed, img_normed, img_normed, img_normed), axis = 0), "label": label}
        else:
            return {"image": img_normed, "label": label}

    def __len__(self):
        return len(self.imgs_ids)


