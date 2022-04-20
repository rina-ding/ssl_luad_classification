
import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import torchvision
from model_simsiam import SimSiam
from dataloader_simsiam import DataProcessor
import torch.nn.functional as F
from itertools import cycle
import matplotlib.pyplot as plt
from scipy import interp
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix as cm
import os
from glob import glob

FIGURE_DIR = 'figures'
MODEL_DIR = 'saved_models'
class TrainModel:
    def __init__(self, num_epochs, batch_size):
        self.epochs = num_epochs
        self.batch = batch_size
        self.device = self._get_device()

    def _get_device(self):
        train_on_gpu = torch.cuda.is_available()
        if not train_on_gpu:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
        return device

    def _get_default_transforms(self):
        print('Data augmentation')
        my_transforms = transforms.Compose([transforms.ToPILImage(), 
                                            transforms.ToTensor(),
                                        transforms.RandomResizedCrop(224, scale=(0.2, 1)), 
                                        transforms.RandomApply([ transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                                        transforms.RandomGrayscale(p=0.2),
                                        transforms.GaussianBlur(kernel_size = 221),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),

                                        ])
        return my_transforms

    def start_training(self, path_to_train, path_to_valid, transformation, lr_rate):
        if transformation == 'default':
            train_dataset = DataProcessor(imgs_dir=path_to_train, transformations=self._get_default_transforms())
            valid_dataset = DataProcessor(imgs_dir=path_to_valid, transformations=self._get_default_transforms())
        else:
            train_dataset = DataProcessor(imgs_dir=path_to_train, transformations=None)
            valid_dataset = DataProcessor(imgs_dir=path_to_valid, transformations=None)

        print("="*40)
        print("Images for Training:", len(train_dataset))
        print("Images for Validation:", len(valid_dataset))
        print("="*40)
        trainloader = DataLoader(train_dataset, batch_size=self.batch, shuffle=True, drop_last=False)
        validloader = DataLoader(valid_dataset, batch_size=self.batch, shuffle=True, drop_last=False)
        
        # Instantiate model and other parameters
       
        model = SimSiam().to(self.device)
        lr_rate = lr_rate * self.batch / 256
        optimizer = optim.Adam(model.parameters(), lr=lr_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
        # Varibles to track
        train_losses, val_losses = [], [], []
        valid_loss_min = np.Inf
        tolerance_count = 0


        # Training loop
        epoch = 0
        while tolerance_count <= 5 and epoch <= self.epochs:
            epoch += 1
            print('Epoch ', epoch)
            running_train_loss, running_val_loss = 0.0, 0.0
            epoch_loss = []
            # Put model on train mode
            model.train()
            for data in trainloader:
                image1, image2 = data['image1'].to(self.device, dtype=torch.float), data['image2'].to(self.device, dtype=torch.float)
                optimizer.zero_grad()
                data_dict = model(image1, image2)
                loss = data_dict['loss'].mean() 
                loss.backward()
                optimizer.step()
                running_train_loss += float(loss.item()) 
                epoch_loss.append(float(loss.item() * image1.size(0)))

            # Validation loop
            with torch.no_grad():
                model.eval()
                for data in validloader:
                    image1, image2 = data['image1'].to(self.device, dtype=torch.float), data['image2'].to(self.device, dtype=torch.float)
                        # output = model(images)
                    data_dict = model(image1, image2)
                    loss = data_dict['loss']
                    print('Single val loss ', loss)
                    running_val_loss += float(loss.item())
                    print(loss.item())
                    print(image1.size(0))

            # Calculate average losses
            avg_train_loss = running_train_loss / len(trainloader)
            avg_val_loss = running_val_loss / len(validloader)
            scheduler.step(avg_val_loss)
            
            # Append losses and track metrics
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
        
            # Print results
            print("Epoch:{}/{} - Training Loss:{:.6f} | Validation Loss: {:.6f}".format(
                epoch, self.epochs, avg_train_loss, avg_val_loss))

            # Save model
            if avg_val_loss <= valid_loss_min:
                print("Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...".format(valid_loss_min, avg_val_loss))
                print("-" * 40)
                model_to_save = model.backbone
                model_to_save = torch.nn.Sequential(*(list(model_to_save.children())[:-1])) # Removing the FC layer
                torch.save(model_to_save.state_dict(), os.path.join(MODEL_DIR, 'self_trained.pth'))
                print(model_to_save)
                # Update minimum loss
                valid_loss_min = avg_val_loss
                tolerance_count = 0
                
            elif avg_val_loss > valid_loss_min:
                tolerance_count += 1

            # Save plots
            plt.plot(train_losses, label='Training loss')
            plt.plot(val_losses, label='Validation loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(os.path.join(FIGURE_DIR, 'losses.png'))
            plt.clf()

if __name__ == "__main__":
    # Hyper-param
    num_epcohs = 100
    batches = 128
    learning_rate = 0.01
    train_images = 'train_images'
    valid_images = 'valid_images'
    train_obj = TrainModel(num_epcohs, batches)
    train_obj.start_training(train_images, valid_images, transformation = 'default', learning_rate = learning_rate)
