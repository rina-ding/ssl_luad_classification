# ssl_luad_classification
Self-Supervised Learning for Histopathologic Subtype Classification

In computational pathology, fully-supervised convolutional neural networks have been shown to perform well on tasks such as histology segmentation and classification but require large amounts of expert-annotated labels. In this work, we propose a self-supervised learning pretext task that utilizes the multi-resolution nature of whole slide images to reduce labeling effort. Given a pair of image tiles cropped at different magnification levels, our model predicts whether one tile is contained in the other. We hypothesize that this task induces the model to learn to distinguish different structures presented in the images and thus benefit the downstream classification. The potential of our method was shown in downstream classification of lung adenocarcinomas histologic subtypes using H\&E-images from the National Lung Screening Trial.

<img width="823" alt="figure1" src="https://user-images.githubusercontent.com/30850467/164292299-4d5e34ed-8a1d-40d8-8c72-85e257365e9e.png">


## Directory structure

```
ssl_proposed: contains the implementation for proposed pretext task
ssl_magnification_level_prediction: contains the implementation for an existing pretext task that predicts the magnification level of an image tile
ssl_simsiam: contains the implementation for one of the state-of-the-art contrastive learning method SimSiam
downstream: contains the implementation for the downstream classification task
```

## Model parameters

```
Proposed: 
  learning rate = 0.0001, 
  batch size = 32, 
  dropout = 0.2, 
  
Tile magnification level prediction: 
  learning rate = 0.0001, 
  batch size = 32, 
  dropout = 0.3, 
  
Simsiam: 
  learning rate = 0.01 * batch size / 256 = 0.005, 
  batch size = 128, 

Downstream:
  learning rate = 0.0001, 
  batch size = 16, 
  dropout = 0.2, 

For all: 
  learning rate reduces by 0.1 if validation loss does not decrease for 5 epochs,
  optimizer = Adam with weight decay 1e-4, 
  early stopping patience = 5 epochs
```
