# ssl_luad_classification
Self-Supervised Learning for Histopathologic Subtype Classification

Lung adenocarcinoma (LUAD) is a morphologically heterogeneous disease with five different histologic subtypes. Fully supervised convolutional neural networks can improve the accuracy and reduce subjectivity of LUAD histologic subtyping using hematoxylin and eosin (H&E)-stained whole slide images (WSIs). However, the annotation process is time consuming and labor intensive. In this work, we propose three self-supervised learning (SSL) pretext tasks to reduce labeling effort. These tasks not only leverage the multi-resolution nature of the H&E WSIs but also explicitly consider the relevance to the downstream task of classifying the LUAD histologic subtypes. Two of the tasks involve predicting the spatial relationship between tiles cropped from lower and higher magnification WSIs. We hypothesize that these tasks induce the model to learn to distinguish different tissue structures presented in the images, thus benefiting the downstream classification. The third task involves predicting the eosin stain from the hematoxylin stain, inducing the model to learn cytoplasmic features which are relevant to LUAD subtypes. The effectiveness of the three proposed SSL tasks and their ensemble was demonstrated by comparison with other state-of-the-art pretraining and SSL methods using three publicly available datasets. Our work can be extended to any other cancer types where tissue architectural information is important.

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
