from sklearn.metrics import precision_score, accuracy_score, recall_score, \
    roc_curve, roc_auc_score, auc
import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np
# from keras.utils import np_utils
import torchvision.transforms as transforms
import torchvision
from model import MagLevelModel
from dataloader import DataProcessor
import torch.nn.functional as F
from itertools import cycle
import matplotlib.pyplot as plt
from scipy import interp
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import ConfusionMatrixDisplay
import os
from glob import glob
# from mlxtend.plotting import plot_confusion_matrix

MODEL_DIR = 'saved_models'
FIGURE_DIR = 'figures'
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
        print('Data augmentation')
        my_transforms = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), 
                                            transforms.Resize(size = (224, 224)),
                                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                            ])

        return my_transforms

    def start_training(self, path_to_train, path_to_valid, transformation, lr_rate):
        if transformation == 'default':
            train_dataset = DataProcessor(imgs_dir=path_to_train, transformations=self._get_default_transforms())
            valid_dataset = DataProcessor(imgs_dir=path_to_valid, transformations=self._get_default_transforms())
        else:
            train_dataset = DataProcessor(imgs_dir=path_to_train, transformations=transformation)
            valid_dataset = DataProcessor(imgs_dir=path_to_valid, transformations=transformation)

        print("="*40)
        print("Images for Training:", len(train_dataset))
        print("Images for Validation:", len(valid_dataset))
        print("="*40)
        trainloader = DataLoader(train_dataset, batch_size=self.batch, shuffle=True, drop_last=False)
        validloader = DataLoader(valid_dataset, batch_size=self.batch, shuffle=True, drop_last=False)
        
        # Instantiate model and other parameters
        model = MagLevelModel(self.num_classes).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr_rate, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)
        # Varibles to track
        train_losses, val_losses = [], []
        metrics = {'accuracy': {0: [], 1: [], 2: [], 3: []},
                   'sensitivity': {0: [], 1: [], 2: [], 3: []},
                   'specificity': {0: [], 1: [], 2: [], 3: []}
                   }
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
                images, labels = data['image'].to(self.device, dtype=torch.float), data['label']
                optimizer.zero_grad()
                output = model.forward(images)
                loss = criterion(output.to(self.device), labels.to(self.device)) 
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
                    images, labels = data['image'].to(self.device, dtype=torch.float), data['label']
                    output = model.forward(images) 
                    loss = criterion(output.to(self.device), labels.to(self.device)) 
                    running_val_loss += float(loss.item()) * images.size(0)
                    output_pb = F.softmax(output.cpu(), dim=1)
                    top_ps, top_class = output_pb.topk(1, dim=1)
                    y_prediction.extend(list(top_class.flatten().numpy()))
                    y_truth.extend(list(labels.cpu().flatten().numpy()))
                    scores.extend(output_pb.numpy().tolist())

            # Calculate average losses
            avg_train_loss = running_train_loss / len(trainloader)
            avg_val_loss = running_val_loss / len(validloader)
            cnf_matrix = cm(y_truth, y_prediction, labels=[0, 1, 2, 3])
            print(cnf_matrix)
            
            # Compute evaluations
            FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
            FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
            TP = np.diag(cnf_matrix)
            TN = cnf_matrix.sum() - (FP + FN + TP)
            # Convert to float
            f_p = FP.astype(float)
            f_n = FN.astype(float)
            t_p = TP.astype(float)
            t_n = TN.astype(float)

            # Calculate metrics
            accuracy = (t_p + t_n) / (f_p + f_n + t_p + t_n)
            recall_sensitivity = t_p / (t_p + f_n)
            specificity = t_n / (t_n + f_p)
            precision = t_p / (t_p + f_p)
            f1_score = 2 * (recall_sensitivity * precision / (recall_sensitivity + precision))

            # Append losses and track metrics
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            for index in range(4):
                metrics['accuracy'][index].append(accuracy[index])
                metrics['sensitivity'][index].append(recall_sensitivity[index])
                metrics['specificity'][index].append(specificity[index])
            # aucs.append(model_auc)

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
            plt.legend(frameon=False)
            plt.savefig(os.path.join(FIGURE_DIR, 'losses.png'))
            plt.clf()

            plt.plot(metrics["accuracy"][0], label='5x')
            plt.plot(metrics["accuracy"][1], label='10x')
            plt.plot(metrics["accuracy"][2], label='20x')
            plt.plot(metrics["accuracy"][3], label='40x')

            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.legend(frameon=False)
            plt.savefig(os.path.join(FIGURE_DIR, 'accuracy.png'))
            plt.clf()

            plt.plot(metrics["sensitivity"][0], label='5x')
            plt.plot(metrics["sensitivity"][1], label='10x')
            plt.plot(metrics["sensitivity"][2], label='20x')
            plt.plot(metrics["sensitivity"][3], label='40x')
            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.legend(frameon=False)
            plt.savefig(os.path.join(FIGURE_DIR, 'sensitivity.png'))
            plt.clf()

            plt.plot(metrics["specificity"][0], label='5x')
            plt.plot(metrics["specificity"][1], label='10x')
            plt.plot(metrics["specificity"][2], label='20x')
            plt.plot(metrics["specificity"][3], label='40x')
            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.legend(frameon=False)
            plt.savefig(os.path.join(FIGURE_DIR, 'specificity.png'))
            plt.clf()


if __name__ == "__main__":
    # Hyper-param
    num_epochs = 100
    batches = 32
    learning_rate = 0.0001
    num_classes = 4
    train_images = 'train_images'
    valid_images = 'valid_images'
    train_obj = TrainModel(num_classes, num_epochs, batches)
    train_obj.start_training(train_images, valid_images, transformation = 'default', learning_rate = learning_rate)
