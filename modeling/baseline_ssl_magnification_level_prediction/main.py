import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from model import ModifiedResNet
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
    def __init__(self, num_classes, num_epochs, batch_size, learning_rate):
        self.num_classes = num_classes
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
        custom_transforms = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), 
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
         ])
        return custom_transforms

    def start_training(self, path_to_train, path_to_valid, train_from_interrupted_model):
        train_dataset = DataProcessor(imgs=path_to_train, transformations=self._get_default_transforms())
        valid_dataset = DataProcessor(imgs=path_to_valid, transformations=self._get_default_transforms())

        print("="*40)
        print("Images for Training:", len(train_dataset))
        print("Images for Validation:", len(valid_dataset))
        print("="*40)
        trainloader = DataLoader(train_dataset, batch_size=self.batch, shuffle=True, drop_last=False, num_workers = 1)
        validloader = DataLoader(valid_dataset, batch_size=self.batch, shuffle=True, drop_last=False, num_workers = 1)
        
        # Instantiate model and other parameters
        model = ModifiedResNet(self.num_classes).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
         # Varibles to track
        train_losses, val_losses, aucs = [], [], []
        metrics = {'accuracy': {key: [] for key in range(self.num_classes)},
                   'sensitivity': {key: [] for key in range(self.num_classes)},
                   'specificity': {key: [] for key in range(self.num_classes)},
                   'f1': {key: [] for key in range(self.num_classes)},
                   }
    
        valid_loss_min = np.Inf
        early_stopping_count = 0

        # Training loop
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
            metrics['f1'] = checkpoint['val_f1_all']

        while early_stopping_count <= 10 and epoch < self.epochs:
            print("-"*40)
            print('Epoch ', epoch)
            running_train_loss, running_val_loss = 0.0, 0.0
            epoch_loss = []
            # Put model on train mode
            model.train()
            for data in trainloader:
                images, labels = data['image'].to(self.device, dtype=torch.float), data['label'].to(self.device, dtype=torch.long)
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output.to(self.device), labels) 
                loss.backward()
                optimizer.step()
                running_train_loss += float(loss.item()) * images.size(0)
                epoch_loss.append(float(loss.item() * images.size(0)))

            # Validation loop
            with torch.no_grad():
                model.eval()
                y_truth, y_prediction, scores = [], [], []
                for data in validloader:
                    images, labels = data['image'].to(self.device, dtype=torch.float), data['label'].to(self.device, dtype=torch.long)
                    output = model(images) 
                    loss = criterion(output.to(self.device), labels) 
                    running_val_loss += float(loss.item()) * images.size(0)
                    output_pb = F.softmax(output.cpu(), dim=1)
                    top_ps, top_class = output_pb.topk(1, dim=1)
                    y_prediction.extend(list(top_class.flatten().numpy()))
                    y_truth.extend(list(labels.cpu().flatten().numpy()))
                    scores.extend(output_pb.numpy().tolist())

            # Calculate average losses
            avg_train_loss = running_train_loss / len(trainloader)
            avg_val_loss = running_val_loss / len(validloader)
            scheduler.step(avg_val_loss)
            cnf_matrix = cm(y_truth, y_prediction, labels=range(self.num_classes))
            # print(cnf_matrix)
            
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
            for index in range(self.num_classes):
                metrics['accuracy'][index].append(accuracy[index])
                metrics['sensitivity'][index].append(recall_sensitivity[index])
                metrics['specificity'][index].append(specificity[index])
                metrics['f1'][index].append(f1_score[index])
                
            # Print results
            print("Epoch:{}/{} - Training Loss:{:.6f} | Validation Loss: {:.6f}".format(
                epoch, self.epochs, avg_train_loss, avg_val_loss))
            print("Accuracy:{}\nF1:{}\nSensitivity:{}\nSpecificity:{}".format(
                accuracy, f1_score, recall_sensitivity, specificity))

            # Save model
            if avg_val_loss <= valid_loss_min:
                print("Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...".format(valid_loss_min, avg_val_loss))
                print("-" * 40)
                # Update minimum loss
                valid_loss_min = avg_val_loss
                early_stopping_count = 0
                
                model_to_save = model.backbone
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
                'early_stopping_count': early_stopping_count,
                'val_f1_all':  metrics['f1']
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

            count = 0
            for i in range(self.num_classes):
                plt.plot(metrics["f1"][i], label='class' + str(i+1))
                plt.ylabel('Validation f1 score')
                if (i+1) % 6 == 0:
                    count += 1
                    plt.legend()
                    plt.savefig(os.path.join(FIGURE_DIR, 'f1_score' + str(count) + '.png'))
                    plt.clf()

            plt.clf()
            epoch += 1

if __name__ == "__main__":
    # Hyper-param
    num_epcohs = 200
    batches = 32
    learning_rate = 0.0001
    num_classes = 4
    train_images = 'path_to_train_images'
    valid_images = 'path_to_validation_images'
    train_obj = TrainModel(num_classes, num_epcohs, batches)
    train_obj.start_training(train_images, valid_images, train_from_interrupted_model = False)
