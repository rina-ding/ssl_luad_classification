# ssl_luad_classification
Self-Supervised Learning for Histopathologic Subtype Classification

![overview](overview.png)

## Directory structure

```
ssl_proposed: contains the implementation for proposed pretext task
ssl_magnification_level_prediction: contains the implementation for an existing pretext task that predicts the magnification level of an image tile
ssl_simsiam: contains the implementation for one of the state-of-the-art contrastive learning method SimSiam
downstream: contains the implementation for the downstream classification task
preprocessing: code to generate proposed image pairs of low and high magnification levels from whole slide images
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
  100 epochs, early stopping patience = 5 epochs
```
