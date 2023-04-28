from sklearn.metrics import precision_score, accuracy_score, recall_score, \
    roc_curve, roc_auc_score, auc
import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np
# from keras.utils import np_utils
import torchvision.transforms as transforms
import torchvision
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
from model import ModifiedResNet, ResNetEncoder

FIGURE_DIR = 'path_to_figures'
if not os.path.exists(FIGURE_DIR):
    os.makedirs(FIGURE_DIR)

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
        # Note: Pytorch transforms works mostly on PIL Image so we convert it to that format and then apply
        # transformations. Also, normalize transforms should be applied when the images are converted to tensors.
        my_transforms = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), 
        transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30), 
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        return my_transforms

    def plot_cnf_matrix(self, fold_index, figures_dir, cm, classes,
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
        plt.savefig(os.path.join(figures_dir, 'confusion_matrix_test_set' + str(fold_index) + '.png'))
        plt.clf()

    def runtestset(self, path_to_images, ssl1_model_path, ssl2_model_path,ssl3_model_path, fold_index):
        figures_dir = FIGURE_DIR
        if not os.path.exists(figures_dir):
            os.makedirs(figures_dir)
        if path_to_images:
            dataset = DataProcessor(imgs_dir=path_to_images, transformations=transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Resize((224, 224)), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]))
            print("Images for testing:", len(dataset))
            testloader = DataLoader(dataset, batch_size=self.batch, shuffle=False, drop_last=False)
            # Define model
            pass_one_batch = False
             # Instantiate model 
            model1 = ModifiedResNet(self.num_classes).to(self.device)
            weights = torch.load(os.path.join(ssl1_model_path, 'resnet18_fold' + str(fold_index) + '.pth'), map_location=self.device)
            model1.load_state_dict(weights)

            model2 = ModifiedResNet(self.num_classes).to(self.device)
            weights = torch.load(os.path.join(ssl2_model_path, 'resnet18_fold' + str(fold_index) + '.pth'), map_location=self.device)
            model2.load_state_dict(weights)

            model3 = ResNetEncoder(self.num_classes).to(self.device)
            weights = torch.load(os.path.join(ssl3_model_path, 'resnet18_fold' + str(fold_index) + '.pth'), map_location=self.device)
            model3.load_state_dict(weights)

            # Make Predictions
            with torch.no_grad():
                model1.eval()
                model2.eval()
                model3.eval()
                if pass_one_batch:
                    pass
                else:
                    y_truth, y_prediction, scores = [], [], []
                    for data in testloader:
                        image_combined, image_single, labels = data['image_combined'].to(self.device, dtype=torch.float), data['image_single'].to(self.device, dtype=torch.float), data['label'].to(self.device, dtype=torch.long)
                        output1 = model1.forward(image_combined)
                        output_pb1 = F.softmax(output1.cpu(), dim=1)
                        output_pb1 = output_pb1.numpy().tolist()
                        # top_ps, top_class1 = output_pb1.topk(1, dim=1)

                        output2 = model2.forward(image_combined)
                        output_pb2 = F.softmax(output2.cpu(), dim=1)
                        output_pb2 = output_pb2.numpy().tolist()
                        # top_ps, top_class2 = output_pb2.topk(1, dim=1)

                        output3 = model3.forward(image_single)
                        output_pb3 = F.softmax(output3.cpu(), dim=1)
                        output_pb3 = output_pb3.numpy().tolist()
                        # top_ps, top_class3 = output_pb3.topk(1, dim=1)

                        output_pb_ensemble = []
                        for i in range(len(output_pb1)):
                            pb_one_patient_ssl1 = output_pb1[i]
                            pb_one_patient_ssl2 = output_pb2[i]
                            pb_one_patient_ssl3 = output_pb3[i]
                            pb_one_patient_updated = []
                            for j in range(len(pb_one_patient_ssl1)):
                                if j == 0: # Probability for lepidic
                                    pb_one_patient_updated.append(pb_one_patient_ssl1[j] * 0.6 + pb_one_patient_ssl2[j] * 0.2 + pb_one_patient_ssl3[j] * 0.2)
                                elif j == 1: # Probability for acinar
                                    pb_one_patient_updated.append(pb_one_patient_ssl1[j] * 0.2 + pb_one_patient_ssl2[j] * 0.2 + pb_one_patient_ssl3[j] * 0.6)
                                elif j == 2: # Probability for pap
                                    pb_one_patient_updated.append(pb_one_patient_ssl1[j] * 0.6 + pb_one_patient_ssl2[j] * 0.2 + pb_one_patient_ssl3[j] * 0.2)
                                elif j == 3: # Probability for micropap
                                    pb_one_patient_updated.append(pb_one_patient_ssl1[j] * 0.6 + pb_one_patient_ssl2[j] * 0.2 + pb_one_patient_ssl3[j] * 0.2)
                                elif j == 4: # Probability for solid
                                    pb_one_patient_updated.append(pb_one_patient_ssl1[j] * 0.2 + pb_one_patient_ssl2[j] * 0.2 + pb_one_patient_ssl3[j] * 0.6)
                                elif j == 5: # Probability for nontumor
                                    pb_one_patient_updated.append(pb_one_patient_ssl1[j] * 0.2 + pb_one_patient_ssl2[j] * 0.2 + pb_one_patient_ssl3[j] * 0.6)

                                
                            output_pb_ensemble.append(pb_one_patient_updated)
                        
                        top_ps, top_class = torch.from_numpy(np.asarray(output_pb_ensemble)).topk(1, dim=1)
                        y_prediction.extend(list(top_class.flatten().numpy()))
                        y_truth.extend(list(labels.cpu().flatten().numpy()))

                    # Computer metrics
                    cnf_matrix = cm(y_truth, y_prediction, labels = [0, 1, 2, 3, 4, 5])
                    print(cnf_matrix)
                    self.plot_cnf_matrix(fold_index, figures_dir, cnf_matrix, ['lepidic', 'acinar', 'pap', 'micropap', 'solid', 'nontumor'],
                            normalize=False,
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

                    print("Accuracy:{}\nPrecision:{}\nSensitivity:{}\nSpecificity:{}\nF1:{}".format(accuracy,
                                                                                                        precision, recall_sensitivity, specificity, f1_score))
        return f1_score

if __name__ == "__main__":
    # Hyper-param
    num_epcohs = 200
    batches = 1
    num_classes = 6
    test_f1_scores = []
    for fold_index in range(5):
        test_images = 'path_to_test_images'
        ssl1_model_path = 'ssl1_model_weights'
        ssl2_model_path = 'ssl2_model_weights'
        ssl3_model_path = 'ssl3_model_weights'

        train_obj = TrainModel(num_classes, num_epcohs, batches, fold_index)
        test_f1_score = train_obj.runtestset(test_images, ssl1_model_path, ssl2_model_path,ssl3_model_path, fold_index)
        test_f1_scores.append(test_f1_score)
    
    test_f1_scores = np.asarray(test_f1_scores)
    print('All folds f1 scores ', test_f1_scores, 3)
    print('Average test f1 scores ', np.nanmean(test_f1_scores, axis = 0))
    print('Standard deviation test f1 scores ', np.nanstd(test_f1_scores, axis = 0))
