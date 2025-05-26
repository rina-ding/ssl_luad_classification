
import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from dataloader import DataProcessor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
from model import EncoderDecoderUNET
from PIL import Image
import argparse
from glob import glob


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

    def _transform(self):
        my_transforms = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor(), 
        ])
        
        return my_transforms

    def start_training(self, path_to_train, path_to_valid, inference_images, train_from_interrupted_model):
        train_dataset = DataProcessor(imgs_dir=path_to_train,  transformations=self._transform())
        valid_dataset = DataProcessor(imgs_dir=path_to_valid, transformations=self._transform())

        print("="*40)
        print("Images for Training:", len(train_dataset))
        print("Images for Validation:", len(valid_dataset))
        print("="*40)
        trainloader = DataLoader(train_dataset, batch_size=self.batch, shuffle=True, drop_last=False, num_workers = 4)
        validloader = DataLoader(valid_dataset, batch_size=self.batch, shuffle=True, drop_last=False, num_workers = 4)
        
        if inference_images:
            inference_dataset = DataProcessor(imgs_dir=inference_images, transformations=self._transform())
            inferenceloader = DataLoader(inference_dataset, batch_size=self.batch, shuffle=False, drop_last=False)

        # Instantiate model and other parameters
        model = EncoderDecoderUNET(3, 3).to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-4)

        criterion = nn.L1Loss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
        # Varibles to track
        train_losses, val_losses, aucs = [], [], []
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

        # Training loop
        while early_stopping_count < 10 and epoch < self.epochs:
            print("-"*40)
            epoch += 1
            print('Epoch ', epoch)
            running_train_loss, running_val_loss = 0.0, 0.0
            epoch_loss = []
            # Put model on train mode
            model.train()
            for data in trainloader:
                he_color, h_stain_image, e_stain_image = data['he_color'].to(self.device, dtype=torch.float), data['h_stain_image'].to(self.device, dtype=torch.float), data['e_stain_image'].to(self.device, dtype=torch.float)
                optimizer.zero_grad()
                output_pred_e = model(h_stain_image)
                loss = criterion(output_pred_e, e_stain_image)
                loss.backward()
                optimizer.step()
                running_train_loss += float(loss.item()) * he_color.size(0)
                epoch_loss.append(float(loss.item() * he_color.size(0)))

            # Validation loop
            with torch.no_grad():
                model.eval()
                for data in validloader:
                    he_color, h_stain_image, e_stain_image = data['he_color'].to(self.device, dtype=torch.float), data['h_stain_image'].to(self.device, dtype=torch.float), data['e_stain_image'].to(self.device, dtype=torch.float)
                    output_pred_e = model(h_stain_image)
                    loss = criterion(output_pred_e, e_stain_image)
                    running_val_loss += float(loss.item()) * he_color.size(0)
                    he_color = (np.asarray(data['he_color'][0]).transpose(1, 2, 0) * 255).astype(np.uint8)
                    h_img = (np.asarray(data['h_stain_image'][0]).transpose(1, 2, 0) * 255).astype(np.uint8)
                    e_img = (np.asarray(data['e_stain_image'][0]).transpose(1, 2, 0) * 255).astype(np.uint8)
                    e_pred = np.asarray(output_pred_e[0].cpu()).transpose(1, 2, 0)

                    Image.fromarray(he_color).save(os.path.join(FIGURE_DIR,'he_color.png'))
                    Image.fromarray(h_img).save(os.path.join(FIGURE_DIR,'h.png'))
                    Image.fromarray(e_img).save(os.path.join(FIGURE_DIR,'e.png'))

                    p = np.percentile(e_pred, 90)
                    e_pred = np.clip(e_pred * 255.0 / p, 0, 255).astype(np.uint8)
                    Image.fromarray(e_pred).save(os.path.join(FIGURE_DIR,'e_pred.png'))
            
                # Inference loop
                if inference_images:
                    for data in inferenceloader:
                        he_color, h_stain_image, e_stain_image = data['he_color'].to(self.device, dtype=torch.float), data['h_stain_image'].to(self.device, dtype=torch.float), data['e_stain_image'].to(self.device, dtype=torch.float)
                        output_pred_e = model(h_stain_image)

                        for i in range(len(he_color)):
                            he_color = (np.asarray(data['he_color'][i]).transpose(1, 2, 0) * 255).astype(np.uint8)
                            h_img = (np.asarray(data['h_stain_image'][i]).transpose(1, 2, 0) * 255).astype(np.uint8)
                            e_img = (np.asarray(data['e_stain_image'][i]).transpose(1, 2, 0) * 255).astype(np.uint8)
                            e_pred = np.asarray(output_pred_e[i].cpu()).transpose(1, 2, 0)

                            output_dir = os.path.join(FIGURE_DIR, 'inference_results')
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)

                            Image.fromarray(he_color).save(os.path.join(output_dir, 'he_color.png'))
                            Image.fromarray(h_img).save(os.path.join(FIGURE_DIR,'h.png'))
                            Image.fromarray(e_img).save(os.path.join(output_dir, 'e.png'))

                            p = np.percentile(e_pred, 90)
                            e_pred = np.clip(e_pred * 255.0 / p, 0, 255).astype(np.uint8)
                            Image.fromarray(e_pred).save(os.path.join(output_dir, 'e_pred.png'))

                    
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
                model_to_save_for_downstream = model.unet.encoder
                torch.save(model_to_save_for_downstream.state_dict(), os.path.join(MODEL_DIR, 'ssl3_trained_model.pth'))
                
                model_complete = model
                torch.save({
                'epoch': epoch,
                'model_state_dict': model_complete.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict' : scheduler.state_dict(),
                'train_loss_all': train_losses,
                'val_loss_all': val_losses,
                'valid_loss_min':valid_loss_min,
                'early_stopping_count':early_stopping_count
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

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_train_images', type = str, default = None, help = 'parent path to the training images')
    parser.add_argument('--path_to_val_images', type = str, default = None, help = 'parent path to the validation images')
    parser.add_argument('--path_to_inference_images', type = str, default = None, help = 'parent path to the inference images')

    parser.add_argument('--num_epochs', type = int, default = 200, help = 'the maximum number of training epochs')
    parser.add_argument('--batches', type = int, default = 32, help = 'batch size')
    parser.add_argument('--learning_rate', type = float, default = 1e-4, help = 'learning rate')
    parser.add_argument('--train_from_interrupted_model', type = bool, default = False, help = 'whether to train model from previously saved complete checkpoints')

    args = parser.parse_args()

    num_epcohs = args.num_epochs
    batches = args.batches
    learning_rate = args.learning_rate
    
    train_images = args.path_to_train_images
    val_images = args.path_to_val_images
    inference_images = args.path_to_inference_images


    FIGURE_DIR = './saved_figures'
    MODEL_DIR = './saved_models'
    if not os.path.exists(FIGURE_DIR):
        os.makedirs(FIGURE_DIR)
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    train_obj = TrainModel(num_epcohs, batches, learning_rate)
    train_obj.start_training(train_images, val_images, inference_images, train_from_interrupted_model = False)

    

    
