# ssl_luad_classification
Self-Supervised Learning for Histopathologic Subtype Classification

We propose a pretext task that predicts whether an image cropped at a higher magnification level is contained in another image cropped at a lower magnification level. We hypothesize that this task induces the model to learn to distinguish different structures presented in WSIs, and thus benefit the downstream classification where those structures are also present.

## Directory structure
```
ssl_proposed: contains the implementation for proposed pretext task
ssl_magnification_level_prediction: contains the implementation for an existing pretext task that predicts the magnification level of an image tile
ssl_simsiam: contains the implementation for one of the state-of-the-art contrastive learning method SimSiam
downstream: contains the implementation for the downstream classification task
```
