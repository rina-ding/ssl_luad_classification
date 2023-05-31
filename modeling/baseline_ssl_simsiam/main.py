
import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from model import SimSiam
from dataloader import DataProcessor
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy import interp
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix as cm
import os
from glob import glob

FIGURE_DIR = './saved_figures'
MODEL_DIR = './saved_models'
if not os.path.exists(FIGURE_DIR):
    os.makedirs(FIGURE_DIR)
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
class TrainModel:
    def __init__(self, num_epochs, batch_size, learning_rate):
        self.epochs = num_epochs
        self.batch = batch_size
        self.learning_rate = learning_rate
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

    def start_training(self, path_to_train, path_to_valid, train_from_interrupted_model):
        train_dataset = DataProcessor(imgs_dir=path_to_train, transformations=self._get_default_transforms())
        valid_dataset = DataProcessor(imgs_dir=path_to_valid, transformations=self._get_default_transforms())

        print("="*40)
        print("Images for Training:", len(train_dataset))
        print("Images for Validation:", len(valid_dataset))
        print("="*40)
        trainloader = DataLoader(train_dataset, batch_size=self.batch, shuffle=True, drop_last=False)
        validloader = DataLoader(valid_dataset, batch_size=self.batch, shuffle=True, drop_last=False)
        
        # Instantiate model and other parameters
       
        model = SimSiam().to(self.device)
        lr_rate = self.learning_rate * self.batch / 256
        optimizer = torch.optim.SGD(model.parameters(), lr=lr_rate, momentum = 0.9)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
        # Varibles to track
        train_losses, val_losses = [], []
        valid_loss_min = np.Inf
        early_stopping_count = 0
        epoch = 0

        if train_from_interrupted_model:
            checkpoint = torch.load(os.path.join(MODEL_DIR, 'complete_model.pth'))
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            train_losses = checkpoint['train_loss_all']
            val_losses = checkpoint['val_loss_all']
            valid_loss_min = checkpoint['valid_loss_min']
            early_stopping_count = checkpoint['early_stopping_count']
            epoch = checkpoint['epoch']

        while early_stopping_count <= 10 and epoch <= self.epochs:
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
                    running_val_loss += float(loss.item())

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
                # Update minimum loss
                valid_loss_min = avg_val_loss
                early_stopping_count = 0
                model_to_save = model.backbone
                model_to_save = torch.nn.Sequential(*(list(model_to_save.children())[:-1])) # Removing the FC layer
                torch.save(model_to_save.state_dict(), os.path.join(MODEL_DIR, 'self_trained.pth'))

                model_complete = model
                torch.save({
                'epoch': epoch,
                'model_state_dict': model_complete.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict' : scheduler.state_dict(),
                'train_loss_all': train_losses,
                'val_loss_all': val_losses,
                'valid_loss_min':valid_loss_min,
                'early_stopping_count': early_stopping_count
                }, 
                os.path.join(MODEL_DIR, 'complete_model.pth'))
                
                
            elif avg_val_loss > valid_loss_min:
                early_stopping_count += 1

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
    num_epcohs = 200
    batches = 128
    learning_rate_initial = 0.01
    train_images = 'path_to_train_images'
    valid_images = 'path_to_validation_images' 
    train_obj = TrainModel(num_epcohs, batches, learning_rate_initial)
    train_obj.start_training(train_images, valid_images, train_from_interrupted_model = True)
