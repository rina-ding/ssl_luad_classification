# ssl_luad_classification
Self-Supervised Learning for Histopathologic Subtype Classification

![overview](overview.png)

## Directory structure

```
modeling
  proposed_ssl1: contains the implementation for proposed pretext task 1

  proposed_ssl2: contains the implementation for proposed pretext task 2

  proposed_ssl3: contains the implementation for proposed pretext task 3

  baseline_ssl_magnification_level_prediction: contains the implementation for an existing pretext task that predicts the magnification level of an image tile

  baseline_ssl_jigmag: contains the implementation for an existing pretext task where the model takes in a sequence of image tiles cropped at different magnification levels with various orders and predicts the arrangement of those tiles.

  baseline_ssl_simsiam: contains the implementation for one of the state-of-the-art contrastive learning method SimSiam

  baseline_ssl_byol: contains the implementation for one of the state-of-the-art contrastive learning method BYOL

  downstream: contains the implementation for the downstream classification task

  downstream_ensemble: contains the implementation for the ensemble of downstream classification models initialized with the three proposed SSL methods.

preprocessing: code to generate data from whole slide images
```
