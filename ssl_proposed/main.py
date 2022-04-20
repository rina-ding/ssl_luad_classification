
import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import torchvision
from model import ModifiedResNet
from dataloader import DataProcessor
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
    def __init__(self, num_classes, num_epochs, batch_size):
        self.num_classes = num_classes
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
        custom_transforms = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), 
         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        return custom_transforms

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
        model = ModifiedResNet(self.num_classes).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr_rate, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)
        # Varibles to track
        train_losses, val_losses = [], []
        accuracy_list, sensitivity_list, specificity_list = [], [], []
        valid_loss_min = np.Inf
        tolerance_count = 0

        # Training loop
        epoch = 0
        while tolerance_count <= 5 and epoch <= self.epochs:
            print("-"*40)
            epoch += 1
            print('Epoch ', epoch)
            running_train_loss, running_val_loss = 0.0, 0.0
            epoch_loss = []
            # Put model on train mode
            model.train()
            for data in trainloader:
                images, labels = data['image'].to(self.device, dtype=torch.float), data['label'].to(self.device, dtype=torch.float)
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output.to(self.device), labels) 
                loss.backward()
                optimizer.step()
                running_train_loss += float(loss.item()) * images.size(0)
                epoch_loss.append(float(loss.item() * images.size(0)))
            scheduler.step(np.mean(epoch_loss))

            # Validation loop
            with torch.no_grad():
                model.eval()
                y_truth, y_prediction, scores = [], [], []
                for data in validloader:
                    images, labels = data['image'].to(self.device, dtype=torch.float), data['label'].to(self.device, dtype=torch.float)
                    output = model(images) 
                    loss = criterion(output.to(self.device), labels) 
                    running_val_loss += float(loss.item()) * images.size(0)
                    output_pb = torch.sigmoid(output.cpu())
                    top_class =  (output_pb.flatten() > 0.5)*1
                    y_prediction.extend(list(top_class.flatten().numpy()))
                    y_truth.extend(list(labels.cpu().flatten().numpy()))

            # Calculate average losses
            avg_train_loss = running_train_loss / len(trainloader)
            avg_val_loss = running_val_loss / len(validloader)
            cnf_matrix = cm(y_truth, y_prediction, labels=[0, 1])
            print(cnf_matrix)
            
            # Compute evaluations
            total=sum(sum(cnf_matrix))
            #####from confusion matrix calculate accuracy
            accuracy=(cnf_matrix[0,0]+cnf_matrix[1,1])/total
            recall_sensitivity = cnf_matrix[0,0]/(cnf_matrix[0,0]+cnf_matrix[0,1])
            precision = cnf_matrix[0,0]/(cnf_matrix[0,0]+cnf_matrix[1, 0])
            specificity = cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1])
            f1_score = 2 * (recall_sensitivity * precision / (recall_sensitivity + precision))

            # Append losses and track metrics
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            accuracy_list.append(accuracy)
            sensitivity_list.append(recall_sensitivity)
            specificity_list.append(specificity)

            # Print results
            print("Epoch:{}/{} - Training Loss:{:.6f} | Validation Loss: {:.6f}".format(
                epoch, self.epochs, avg_train_loss, avg_val_loss))
            print("Accuracy:{}\nPrecision:{}\nSensitivity:{}\nSpecificity:{}".format(
                accuracy, precision, recall_sensitivity, specificity))

            # Save model
            if avg_val_loss <= valid_loss_min:
                print("Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...".format(valid_loss_min, avg_val_loss))
                print("-" * 40)
                model_to_save = model.backbone
                torch.save(model_to_save.state_dict(), os.path.join(MODEL_DIR, 'self_trained.pth'))
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

            plt.plot(accuracy_list, label = 'Validation accuracy')        
            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.legend()
            plt.savefig(os.path.join(FIGURE_DIR, 'accuracy.png'))
            plt.clf()

            plt.plot(sensitivity_list, label = 'Validation sensitivity')
            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.legend()
            plt.savefig(os.path.join(FIGURE_DIR, 'sensitivity.png'))
            plt.clf()

            plt.plot(specificity_list, label = 'Validation specificity')
            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.legend()
            plt.savefig(os.path.join(FIGURE_DIR, 'specificity.png'))
            plt.clf()

if __name__ == "__main__":
    # Hyper-param
    num_epcohs = 100
    batches = 32
    learning_rate = 0.0001
    num_classes = 1
    train_images = 'train_images'
    valid_images = 'valid_images'
    train_obj = TrainModel(num_classes, num_epcohs, batches)
    train_obj.start_training(train_images, valid_images, transformation = 'default', learning_rate = learning_rate)
