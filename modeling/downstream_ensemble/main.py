
import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision
from dataloader import DataProcessor
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix as cm
import os
from glob import glob
from model import ModifiedResNet, ResNetEncoder
import argparse

class TrainModel:
    def __init__(self, num_classes, fold_index):
        self.num_classes = num_classes
        self.batch = 1
        self.fold_index = fold_index
        self.device = self._get_device()

    def _get_device(self):
        train_on_gpu = torch.cuda.is_available()
        if not train_on_gpu:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
        return device

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

    def find_best_weights(self, path_to_valid_images, path_to_test_images, ssl1_model_path, ssl2_model_path,ssl3_model_path, fold_index):
        figures_dir = FIGURE_DIR
        if not os.path.exists(figures_dir):
            os.makedirs(figures_dir)
        valid_dataset = DataProcessor(imgs_dir=path_to_valid_images, transformations=transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Resize((224, 224)), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]))
        test_dataset = DataProcessor(imgs_dir=path_to_test_images, transformations=transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Resize((224, 224)), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]))
        
        print("Images for val:", len(valid_dataset))
        print("Images for testing:", len(test_dataset))
        valloader = DataLoader(valid_dataset, batch_size=self.batch, shuffle=False, drop_last=False, num_workers = 4)
        testloader = DataLoader(test_dataset, batch_size=self.batch, shuffle=False, drop_last=False, num_workers = 4)
       
        model1 = ModifiedResNet(self.num_classes).to(self.device)
        weights = torch.load(ssl1_model_path, map_location=self.device)
        model1.load_state_dict(weights)

        model2 = ModifiedResNet(self.num_classes).to(self.device)
        weights = torch.load(ssl2_model_path, map_location=self.device)
        model2.load_state_dict(weights)

        model3 = ResNetEncoder(self.num_classes).to(self.device)
        weights = torch.load(ssl3_model_path, map_location=self.device)
        model3.load_state_dict(weights)
        # Define the weight combinations to be searched
        weights = np.linspace(0, 1, 11)  # 11 evenly spaced values between 0 and 1

        best_weights = None
        best_f1_score = 0
        with torch.no_grad():
            model1.eval()
            model2.eval()
            model3.eval()

            # Perform grid search for the best weights
            for w1 in weights:
                for w2 in weights:
                    for w3 in weights:
                        # Skip weight combinations where the sum is not close to 1
                        if not np.isclose(w1 + w2 + w3, 1):
                            continue
                        print((w1, w2, w3))
                        y_truth, y_prediction, scores = [], [], []
                        for data in valloader:
                            image_combined, image_single, labels = data['image_combined'].to(self.device, dtype=torch.float), data['image_single'].to(self.device, dtype=torch.float), data['label'].to(self.device, dtype=torch.long)
                            output1 = model1.forward(image_combined)
                            output_pb1 = F.softmax(output1.cpu(), dim=1)
                            output_pb1 = output_pb1.numpy().tolist()

                            output2 = model2.forward(image_combined)
                            output_pb2 = F.softmax(output2.cpu(), dim=1)
                            output_pb2 = output_pb2.numpy().tolist()

                            output3 = model3.forward(image_single)
                            output_pb3 = F.softmax(output3.cpu(), dim=1)
                            output_pb3 = output_pb3.numpy().tolist()
                            output_pb_ensemble = []
                            for i in range(len(output_pb1)):
                                pb_one_patient_ssl1 = output_pb1[i]
                                pb_one_patient_ssl2 = output_pb2[i]
                                pb_one_patient_ssl3 = output_pb3[i]
                                pb_one_patient_updated = []
                                for j in range(len(pb_one_patient_ssl1)):
                                    pb_one_patient_updated.append(pb_one_patient_ssl1[j] * w1 + pb_one_patient_ssl2[j] * w2 + pb_one_patient_ssl3[j] * w3)
                                output_pb_ensemble.append(pb_one_patient_updated)

                            top_ps, top_class = torch.from_numpy(np.asarray(output_pb_ensemble)).topk(1, dim=1)
                            y_prediction.extend(list(top_class.flatten().numpy()))
                            y_truth.extend(list(labels.cpu().flatten().numpy()))

                        cnf_matrix = cm(y_truth, y_prediction, labels = [0, 1, 2, 3, 4, 5])
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
                        recall_sensitivity = t_p / (t_p + f_n)
                        precision = t_p / (t_p + f_p)
                        f1_score = 2 * (recall_sensitivity * precision / (recall_sensitivity + precision))
                        # Update best F1 score and weights if necessary
                        if np.mean(f1_score) > best_f1_score:
                            best_f1_score = np.mean(f1_score)
                            best_weights = (w1, w2, w3)
                        
            print("Best weights:", best_weights)
            print("Best F1 score:", best_f1_score)

            # Test loop using test set after selecting the weights using validation set
            y_truth, y_prediction, scores = [], [], []
            for data in testloader:
                image_combined, image_single, labels = data['image_combined'].to(self.device, dtype=torch.float), data['image_single'].to(self.device, dtype=torch.float), data['label'].to(self.device, dtype=torch.long)
                output1 = model1.forward(image_combined)
                output_pb1 = F.softmax(output1.cpu(), dim=1)
                output_pb1 = output_pb1.numpy().tolist()

                output2 = model2.forward(image_combined)
                output_pb2 = F.softmax(output2.cpu(), dim=1)
                output_pb2 = output_pb2.numpy().tolist()

                output3 = model3.forward(image_single)
                output_pb3 = F.softmax(output3.cpu(), dim=1)
                output_pb3 = output_pb3.numpy().tolist()

                output_pb_ensemble = []
                for i in range(len(output_pb1)):
                    pb_one_patient_ssl1 = output_pb1[i]
                    pb_one_patient_ssl2 = output_pb2[i]
                    pb_one_patient_ssl3 = output_pb3[i]
                    pb_one_patient_updated = []
                    for j in range(len(pb_one_patient_ssl1)):
                        pb_one_patient_updated.append(pb_one_patient_ssl1[j] * best_weights[0] + pb_one_patient_ssl2[j] * best_weights[1] + pb_one_patient_ssl3[j] * best_weights[2])
                    output_pb_ensemble.append(pb_one_patient_updated)
                
                top_ps, top_class = torch.from_numpy(np.asarray(output_pb_ensemble)).topk(1, dim=1)
                y_prediction.extend(list(top_class.flatten().numpy()))
                y_truth.extend(list(labels.cpu().flatten().numpy()))

            # Computer metrics
            cnf_matrix = cm(y_truth, y_prediction, labels = [0, 1, 2, 3, 4, 5])
            self.plot_cnf_matrix(fold_index, figures_dir, cnf_matrix, ['lepidic', 'acinar', 'pap', 'micropap', 'solid', 'nontumor'],
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
            recall_sensitivity = t_p / (t_p + f_n)
            precision = t_p / (t_p + f_p)
            f1_score = 2 * (recall_sensitivity * precision / (recall_sensitivity + precision))

        return f1_score, best_weights

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_all_images', type = str, default = None, help = 'parent path to all the images')
    args = parser.parse_args()

    root_dir_for_images = args.path_to_all_images

    FIGURE_DIR = './saved_figures'
    if not os.path.exists(FIGURE_DIR):
        os.makedirs(FIGURE_DIR)
  
    num_classes = 6
    test_f1_scores = []
    best_weights_list = []
    for fold_index in range(5):
        valid_images = glob(os.path.join(root_dir_for_images, 'fold' + str(fold_index), 'val', '*', '*.png'))
        test_images = glob(os.path.join(root_dir_for_images, 'fold' + str(fold_index), 'test', '*', '*.png'))

        ssl1_model_path = os.path.join('./individual_downstream_model_weights', 'proposed_ssl1', 'resnet18_fold' + str(fold_index) + '.pth')
        ssl2_model_path = os.path.join('./individual_downstream_model_weights', 'proposed_ssl2', 'resnet18_fold' + str(fold_index) + '.pth')
        ssl3_model_path = os.path.join('./individual_downstream_model_weights', 'proposed_ssl3', 'resnet18_fold' + str(fold_index) + '.pth')

        train_obj = TrainModel(num_classes, fold_index)
        f1_scores, best_weights = train_obj.find_best_weights(valid_images, test_images, ssl1_model_path, ssl2_model_path,ssl3_model_path, fold_index)
        test_f1_scores.append(f1_scores)
        best_weights_list.append(best_weights)
    
    test_f1_scores = np.asarray(test_f1_scores)
    print('All f1 scores')
    print('', test_f1_scores, 3)
    print('Average test f1 scores ', np.nanmean(test_f1_scores, axis = 0))
    print('Standard deviation test f1 scores ', np.nanstd(test_f1_scores, axis = 0))
    print('Best weights')
    print(best_weights_list)
