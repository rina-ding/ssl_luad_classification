
import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from dataloader import DataProcessor
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix as cm
import os
from glob import glob
from model import ModifiedResNet
import argparse

class TrainModel:
    def __init__(self, num_classes, num_epochs, batch_size, learning_rate, fold_index):
        self.num_classes = num_classes
        self.epochs = num_epochs
        self.batch = batch_size
        self.fold_index = fold_index
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
        my_transforms = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), 
        transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30), 
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        return my_transforms

    def plot_cnf_matrix(self, figures_dir, cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        import itertools
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=25)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(os.path.join(figures_dir, 'confusion_matrix_test_set.png'))
        plt.clf()

    def start_training(self, path_to_train, path_to_valid):
        train_dataset = DataProcessor(imgs_dir=path_to_train, channel = NUM_CHANNEL, transformations=self._get_default_transforms())
        valid_dataset = DataProcessor(imgs_dir=path_to_valid, channel = NUM_CHANNEL, transformations=None)
        
        print("="*40)
        print("Images for Training:", len(train_dataset))
        print("Images for Validation:", len(valid_dataset))
        print("="*40)
        trainloader = DataLoader(train_dataset, batch_size=self.batch, shuffle=True, drop_last=False)
        validloader = DataLoader(valid_dataset, batch_size=self.batch, shuffle=True, drop_last=False)
       
        # Instantiate model and other parameters
        model = ModifiedResNet(self.num_classes, self.device, NUM_CHANNEL).to(self.device)

        # # Self supervised pretraining weights
        weights = torch.load(glob(os.path.join(path_to_ssl_model_weights, '*.pth'))[0], map_location=self.device)
        model.load_state_dict({'backbone.' + k[0:]:v for k, v in weights.items()}, strict=False)
       
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
        # Varibles to track
        train_losses, val_losses, aucs = [], [], []
        metrics = {'accuracy': {0: [], 1: [], 2: [], 3: [], 4: [], 5: []},
                   'sensitivity': {0: [], 1: [], 2: [], 3: [], 4: [], 5: []},
                   'specificity': {0: [], 1: [], 2: [], 3: [], 4: [], 5: []},
                   'f1': {0: [], 1: [], 2: [], 3: [], 4: [], 5: []},
                   }
        valid_loss_min = np.Inf
        early_stopping_count = 0

        # Training loop
        epoch = 0
        while early_stopping_count < 10 and epoch < self.epochs:
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
            scheduler.step(avg_val_loss)
            cnf_matrix = cm(y_truth, y_prediction, labels=[0, 1, 2, 3, 4, 5])
            
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
            for index in range(6):
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
                torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'resnet18' + '_fold' + str(self.fold_index) + '.pth'))
            elif avg_val_loss > valid_loss_min:
                early_stopping_count += 1

        # Save plots
        figures_dir = os.path.join(FIGURE_DIR, 'fold' + str(self.fold_index))
        if not os.path.exists(figures_dir):
            os.makedirs(figures_dir)
        plt.plot(train_losses, label='Training loss')
        plt.plot(val_losses, label='Validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(frameon=False)
        plt.savefig(os.path.join(figures_dir, 'losses.png'))
        plt.clf()

        plt.plot(metrics["accuracy"][0], label='lepidic')
        plt.plot(metrics["accuracy"][1], label='acinar')
        plt.plot(metrics["accuracy"][2], label='papillary')
        plt.plot(metrics["accuracy"][3], label='micropapillary')
        plt.plot(metrics["accuracy"][4], label='solid')
        plt.plot(metrics["accuracy"][5], label='nontumor')

        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend(frameon=False)
        plt.savefig(os.path.join(figures_dir, 'accuracy.png'))
        plt.clf()

        plt.plot(metrics["sensitivity"][0], label='lepidic')
        plt.plot(metrics["sensitivity"][1], label='acinar')
        plt.plot(metrics["sensitivity"][2], label='papillary')
        plt.plot(metrics["sensitivity"][3], label='micropapillary')
        plt.plot(metrics["sensitivity"][4], label='solid')
        plt.plot(metrics["sensitivity"][5], label='nontumor')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend(frameon=False)
        plt.savefig(os.path.join(figures_dir, 'sensitivity.png'))
        plt.clf()

        plt.plot(metrics["specificity"][0], label='lepidic')
        plt.plot(metrics["specificity"][1], label='acinar')
        plt.plot(metrics["specificity"][2], label='papillary')
        plt.plot(metrics["specificity"][3], label='micropapillary')
        plt.plot(metrics["specificity"][4], label='solid')
        plt.plot(metrics["specificity"][5], label='nontumor')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend(frameon=False)
        plt.savefig(os.path.join(figures_dir, 'specificity.png'))
        plt.clf()

        plt.plot(metrics["f1"][0], label='lepidic')
        plt.plot(metrics["f1"][1], label='acinar')
        plt.plot(metrics["f1"][2], label='papillary')
        plt.plot(metrics["f1"][3], label='micropapillary')
        plt.plot(metrics["f1"][4], label='solid')
        plt.plot(metrics["f1"][5], label='nontumor')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend(frameon=False)
        plt.savefig(os.path.join(figures_dir, 'f1.png'))
        plt.clf()

    def runtestset(self, path_to_images=None):
        figures_dir = os.path.join(FIGURE_DIR, 'fold' + str(self.fold_index))
        if not os.path.exists(figures_dir):
            os.makedirs(figures_dir)
        if path_to_images:
            dataset = DataProcessor(imgs_dir=path_to_images, channel = NUM_CHANNEL, transformations=transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Resize((224, 224)), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]))
            print("Images for testing:", len(dataset))
            testloader = DataLoader(dataset, batch_size=self.batch, shuffle=True, drop_last=False)
            # Define model
            pass_one_batch = False
            model = ModifiedResNet(self.num_classes, self.device, NUM_CHANNEL).to(self.device)
            
            criterion = nn.CrossEntropyLoss()
            
            print("="*40)
            print("Model Weights Loaded")
            print("=" * 40)
            weights = torch.load(os.path.join(MODEL_DIR, 'resnet18' + '_fold' + str(self.fold_index) + '.pth'), map_location=self.device)
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
                    cnf_matrix = cm(y_truth, y_prediction, labels = [0, 1, 2, 3, 4, 5])
                    print(cnf_matrix)
                    self.plot_cnf_matrix(figures_dir, cnf_matrix, ['lepidic', 'acinar', 'pap', 'micropap', 'solid', 'nontumor'],
                            normalize=True,
                            title='Confusion matrix',
                            cmap=plt.cm.Blues)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_all_images', type = str, default = None, help = 'parent path to all the images')
    parser.add_argument('--path_to_ssl_model_weights', type = str, default = None, help = 'the folder that contains the path to the SSL-pretrained model')
    parser.add_argument('--num_image_channels', type = int, default = 6, help = 'the number of channels for the images. 6 for SSL1 and 2, 3 for SSL3.')
    parser.add_argument('--num_epochs', type = int, default = 200, help = 'the maximum number of training epochs')
    parser.add_argument('--batches', type = int, default = 32, help = 'batch size')
    parser.add_argument('--learning_rate', type = float, default = 1e-4, help = 'learning rate')

    args = parser.parse_args()

    num_epcohs = args.num_epochs
    batches = args.batches
    learning_rate = args.learning_rate
    root_dir_for_images = args.path_to_all_images
    path_to_ssl_model_weights = args.path_to_ssl_model_weights
    NUM_CHANNEL = args.num_image_channels

    FIGURE_DIR = './saved_figures'
    MODEL_DIR = './saved_models'
    if not os.path.exists(FIGURE_DIR):
        os.makedirs(FIGURE_DIR)
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    num_classes = 6
    test_f1_scores = []
    for fold_index in range(5):
        train_images = glob(os.path.join(root_dir_for_images, 'fold' + str(fold_index), 'train', '*', '*.png'))
        valid_images = glob(os.path.join(root_dir_for_images, 'fold' + str(fold_index), 'val', '*', '*.png'))
        test_images = glob(os.path.join(root_dir_for_images, 'fold' + str(fold_index), 'test', '*', '*.png'))
        train_obj = TrainModel(num_classes, num_epcohs, batches, learning_rate, fold_index)
        train_obj.start_training(train_images, valid_images)
        test_f1_score = train_obj.runtestset(test_images)
        test_f1_scores.append(test_f1_score)
    
    test_f1_scores = np.asarray(test_f1_scores)
    print('All folds f1 scores ', test_f1_scores, 3)
    print('Average test f1 scores ', np.nanmean(test_f1_scores, axis = 0))
    print('Standard deviation test f1 scores ', np.nanstd(test_f1_scores, axis = 0))
