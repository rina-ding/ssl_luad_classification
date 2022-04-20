
import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import torchvision
from dataloader_resnet import DataProcessor
import torch.nn.functional as F
from itertools import cycle
import matplotlib.pyplot as plt
from scipy import interp
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix as cm
import os
from glob import glob
from model_resnet import ModifiedResNet

model_type = 'from_scratch'
MODEL_DIR = os.path.join('saved_models', model_type)
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

class TrainModel:
    def __init__(self, num_classes, num_epochs, batch_size, fold_index):
        self.num_classes = num_classes
        self.epochs = num_epochs
        self.batch = batch_size
        self.fold_index = fold_index
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
        my_transforms = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.RandomHorizontalFlip(p=0.5), transforms.ColorJitter(brightness=0.5),
                                            transforms.RandomRotation(degrees=45), transforms.RandomVerticalFlip(p=0.2),
                                            transforms.Resize((224, 224)), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        
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
        model = ModifiedResNet(self.num_classes).to(self.device)
        # # Self supervised pretraining weights
        weights = torch.load(os.path.join('saved_models', 'self_trained.pth'))
        model.load_state_dict({'backbone.' + k[0:]:v for k, v in weights.items()}, strict=False)
    
        optimizer = optim.Adam(model.parameters(), lr=lr_rate, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
        # Varibles to track
        train_losses, val_losses, aucs = [], [], []
        metrics = {'accuracy': {0: [], 1: [], 2: [], 3: [], 4: []},
                   'sensitivity': {0: [], 1: [], 2: [], 3: [], 4: []},
                   'specificity': {0: [], 1: [], 2: [], 3: [], 4: []},
                   'f1': {0: [], 1: [], 2: [], 3: [], 4: []},
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
                images, labels = data['image'].to(self.device, dtype=torch.float), data['label'].to(self.device, dtype=torch.long)
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, labels)
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
                    loss = criterion(output, labels)
                    running_val_loss += float(loss.item()) * images.size(0)
                    output_pb = F.softmax(output.cpu(), dim=1)
                    top_ps, top_class = output_pb.topk(1, dim=1)
                    y_prediction.extend(list(top_class.flatten().numpy()))
                    y_truth.extend(list(labels.cpu().flatten().numpy()))
                    scores.extend(output_pb.numpy().tolist())

            # Calculate average losses
            avg_train_loss = running_train_loss / len(trainloader)
            avg_val_loss = running_val_loss / len(validloader)
            cnf_matrix = cm(y_truth, y_prediction, labels=[0, 1, 2, 3, 4])
            scheduler.step(avg_val_loss)

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
            for index in range(5):
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
                torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'resnet18' + '_fold' + str(self.fold_index) + '.pth'))
                # Update minimum loss
                valid_loss_min = avg_val_loss
                tolerance_count = 0

            elif avg_val_loss > valid_loss_min:
                tolerance_count += 1

    def runtestset(self, path_to_images=None):
        if path_to_images:
            dataset = DataProcessor(imgs_dir=path_to_images, transformations=transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Resize((224, 224)), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]))
            print("Images for testing:", len(dataset))
            testloader = DataLoader(dataset, batch_size=self.batch, shuffle=True, drop_last=False)
            # Define model
            pass_one_batch = False
            model = ModifiedResNet(self.num_classes).to(self.device)
            
            criterion = nn.CrossEntropyLoss()
            
            print("="*40)
            print("Model Weights Loaded")
            print("=" * 40)
            weights = torch.load(os.path.join(MODEL_DIR, 'resnet18' + '_fold' + str(self.fold_index) + '.pth'))
            model.load_state_dict(weights)
            model.to(self.device)

            # Make Predictions
            with torch.no_grad():
                model.eval()
                if pass_one_batch:
                    pass
                else:
                    y_truth, y_prediction, scores = [], [], []
                    running_test_loss = 0.0
                    for data in testloader:
                        images, labels = data['image'].to(self.device, dtype=torch.float), data['label'].to(self.device, dtype=torch.long)
                        # output = model(images)
                        output = model.forward(images)
                        loss = criterion(output, labels)
                        running_test_loss += float(loss.item()) * images.size(0)
                        output_pb = F.softmax(output.cpu(), dim=1)
                        top_ps, top_class = output_pb.topk(1, dim=1)
                        y_prediction.extend(list(top_class.flatten().numpy()))
                        y_truth.extend(list(labels.cpu().flatten().numpy()))
                        scores.extend(output_pb.numpy().tolist())

                    # Computer metrics
                    avg_test_loss = running_test_loss / len(testloader)
                    cnf_matrix = cm(y_truth, y_prediction, labels = [0, 1, 2, 3, 4])
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

                    print("Test loss:{:.6f}".format(avg_test_loss))
                    print("Accuracy:{}\nPrecision:{}\nSensitivity:{}\nSpecificity:{}\nF1:{}".format(accuracy,
                                                                                                        precision, recall_sensitivity, specificity, f1_score))
                    # Computer FPR, TPR for two classes
                    scores = np.array(scores)
        return f1_score
           


if __name__ == "__main__":
    # Hyper-param
    num_epcohs = 100
    batches = 16
    learning_rate = 0.0001
    num_classes = 5
    test_f1_scores = []
    root = ''
    for fold_index in range(5):
        fold_root_dir = os.path.join(root, 'fold' + str(fold_index))
        train_images = os.path.join(root, 'fold' + str(fold_index), 'train')
        valid_images = os.path.join(root + str(fold_index), 'val')
        test_images = os.path.join(root, 'fold' + str(fold_index), 'test')
        train_obj = TrainModel(num_classes, num_epcohs, batches, fold_index)
        train_obj.start_training(train_images, valid_images, transformation='default', learning_rate = learning_rate)
        test_f1_score = train_obj.runtestset(test_images)
        test_f1_scores.append(test_f1_score)
    
    test_f1_scores = np.asarray(test_f1_scores)
    print('All folds f1 scores ', test_f1_scores)
    print('Average test f1 scores ', np.nanmean(test_f1_scores, axis = 0))
    print('Standard deviation test f1 scores ', np.nanstd(test_f1_scores, axis = 0))
