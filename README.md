# ssl_luad_classification
Tailoring Pretext Tasks to Improve Self-Supervised Learning in Histopathologic Subtype Classification of Lung Adenocarcinomas

Lung adenocarcinoma (LUAD) is a morphologically heterogeneous disease with five different histologic subtypes. Fully supervised convolutional neural networks can improve the accuracy and reduce subjectivity of LUAD histologic subtyping using hematoxylin and eosin (H&E)-stained whole slide images (WSIs). However, the annotation process is time consuming and labor intensive. In this work, we propose three self-supervised learning (SSL) pretext tasks to reduce labeling effort. These tasks not only leverage the multi-resolution nature of the H&E WSIs but also explicitly consider the relevance to the downstream task of classifying the LUAD histologic subtypes. Two of the tasks involve predicting the spatial relationship between tiles cropped from lower and higher magnification WSIs. We hypothesize that these tasks induce the model to learn to distinguish different tissue structures presented in the images, thus benefiting the downstream classification. The third task involves predicting the eosin stain from the hematoxylin stain, inducing the model to learn cytoplasmic features which are relevant to LUAD subtypes. The effectiveness of the three proposed SSL tasks and their ensemble was demonstrated by comparison with other state-of-the-art pretraining and SSL methods using three publicly available datasets. Our work can be extended to any other cancer types where tissue architectural information is important.

![overview](overview.png)

## Instructions 
### Required packages
First, create a pytorch docker container using:
```
docker run  --shm-size=2g --gpus all -it --rm -v /:/workspace -v /etc/localtime:/etc/localtime:ro nvcr.io/nvidia/pytorch:24.02-py3
```
Then install all packages by running the following commands:

```
cd ssl_luad_classification
```

```
chmod +x pip_commands.sh
```
```
./pip_commands.sh
```

More information on the pytorch docker container `nvcr.io/nvidia/pytorch:24.02-py3` can be found here(https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags).


### Preprocessing

- Datasets: Download NLST data from [NLST](https://wiki.cancerimagingarchive.net/display/NLST/NLST+Pathology), download TCGA data from [TCGA-LUAD](https://portal.gdc.cancer.gov/projects/TCGA-LUAD), and download CPTAC data from [CPTAC](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=33948253).

Each dataset's folder structure should be:
```
  ├── <patient_id>                   
  │   ├── <slide_id>   
```

- SSL 1 and 2 data prep: 
```
cd preprocessing
```

```
python get_low_high_magnification_image_pairs.py --wsi_level <wsi_level> --tile_size <tile_size> --downsample_factor <downsample_factor> --path_to_wsi_images <> --path_to_generated_tiles <>
```

An example command for when using WSIs that have 40x objective magnification:
```
python get_low_high_magnification_image_pairs.py --wsi_level 2 --tile_size 512 --downsample_factor 16 --path_to_wsi_images <> --path_to_generated_tiles <>
```

`path_to_wsi_images` is the parent path to the WSIs, structured in the format mentioned above.
`path_to_generated_tiles` is the parent path to the generated tiles.

After running this script, you should get output data in the following structure within your specified `path_to_generated_tiles` (specifically under `/<path_to_generated_tiles>/train_ssl1` for training data and `/<path_to_generated_tiles>/val_ssl1` for SSL1 and similarly for SSL2):

```
  ├── train                   
  │   ├── <path_to_a_low_high_magnification_image_pair>
      │   ├── <path_to_low_magnification_image.png>
      │   ├── <path_to_high_magnification_image.png>
      ├── ...
  ├── val                  
  │   ├── <path_to_a_low_high_magnification_image_pair>
      │   ├── <path_to_low_magnification_image.png>
      │   ├── <path_to_high_magnification_image.png>
      ├── ...

```

- SSL 3 data prep:
```
cd preprocessing
```

First run `generate_tiles.py` to get tiles from the WSIs.
```
python generate_tiles.py --wsi_level <wsi_level> --tile_size <tile_size> --downsample_factor <downsample_factor> --path_to_wsi_images <> --path_to_generated_tiles <>
```

Second, run `stain_separation.py` to get the SSL 3 data.
```
python stain_separation.py ----path_to_input_tiles <> ----path_to_output_tiles
```

where `path_to_input_tiles` should be the same path as the `path_to_generated_tiles` from the `generate_tile.py` step.

After running this script, you should get output data in the following structure within your specified `path_to_output_tiles` (specifically under `/<path_to_output_tiles>/train` for training data and `/<path_to_output_tiles>/val` for validation data):

```
  ├── train                   
  │   ├── <path_to_h_stain_images>
      │   ├── <path_to_image1.png>
      │   ├── <path_to_image2.png>
      │   ├── ...
  │   ├── <path_to_e_stain_images>
      │   ├── <path_to_image1.png>
      │   ├── <path_to_image2.png>
      │   ├── ...
  │   ├── <path_to_he_stain_images>
      │   ├── <path_to_image1.png>
      │   ├── <path_to_image2.png>
      │   ├── ...
  ├── val       
  │   ├── <path_to_h_stain_images>
      │   ├── <path_to_image1.png>
      │   ├── <path_to_image2.png>
      │   ├── ...            
  │   ├── <path_to_e_stain_images>
      │   ├── <path_to_image1.png>
      │   ├── <path_to_image2.png>
      │   ├── ...
  │   ├── <path_to_he_stain_images>
      │   ├── <path_to_image1.png>
      │   ├── <path_to_image2.png>
      │   ├── ...
```


### Modeling

#### For SSL 1:

```
cd modeling/proposed_ssl1
```

```
CUDA_VISIBLE_DEVICES=0 python main.py --path_to_train_images <> --path_to_val_images <>
```

`path_to_train_images` is the parent path to the training images generated from the preprocessing step, specifically under `/<path_to_generated_tiles>/train_ssl1`.

`path_to_val_images` is the parent path to the validation images generated from the preprocessing step, specifically under `/<path_to_generated_tiles>/val_ssl1`.

#### For SSL 2:

```
cd modeling/proposed_ssl2
```

```
CUDA_VISIBLE_DEVICES=0 python main.py --path_to_train_images <> --path_to_val_images <>
```

`path_to_train_images` is the parent path to the training images generated from the preprocessing step, specifically under `/<path_to_generated_tiles>/train_ssl2`.

`path_to_val_images` is the parent path to the validation images generated from the preprocessing step, specifically under `/<path_to_generated_tiles>/val_ssl2`.

#### For SSL 3:

```
cd modeling/proposed_ssl3
```

```
CUDA_VISIBLE_DEVICES=0 python main.py --path_to_train_images <> --path_to_val_images <>
```

`path_to_train_images` is the parent path to the training images generated from the preprocessing step, specifically under `/<path_to_output_tiles>/train`.

`path_to_val_images` is the parent path to the validation images generated from the preprocessing step, specifically under `/<path_to_output_tiles>/val`.

For all SSL models, the optional input arguments are:

`num_epochs`: the maximum number of training epochs

`batches`: batch size

`learning_rate`: learning rate

`train_from_interrupted_model`: whether to train model from previously saved complete checkpoints

- Model weights

[ssl1_trained_model.pth](./modeling/proposed_ssl1/ssl1_trained_model.pth)

[ssl2_trained_model.pth](./modeling/proposed_ssl2/ssl2_trained_model.pth)

[ssl3_trained_model.pth](./modeling/proposed_ssl3/ssl3_trained_model.pth)


#### For downstream modeling

- Step 1: Train a downstream model using a SSL-pretrained model

Since we have 3 SSL-pretrained models, we will train 3 different downstream models.

For each downstream model, do:

```
cd modeling/downstream
```

```
CUDA_VISIBLE_DEVICES=0 python main.py --path_to_all_images <> --path_to_ssl_model_weights <>
```

`path_to_all_images` is the parent path that contains all folds' train, val, and test data, organized in the following structure:

- Downstream data format
For downstream model, the input folder structure should be:
```
  ├── fold0        
  │   ├── train
    │   ├── <path_to_a_folder_containing_one_tissue_class>
        │   ├── <path_to_image1.png>
        │   ├── <path_to_image2.png>
        │   ├── <...>
  │   ├── val
    │   ├── <path_to_a_folder_containing_one_tissue_class>
        │   ├── <path_to_image1.png>
        │   ├── <path_to_image2.png>
        │   ├── <...>
  │   ├── test
    │   ├── <path_to_a_folder_containing_one_tissue_class>
        │   ├── <path_to_image1.png>
        │   ├── <path_to_image2.png>
        │   ├── <...>

  ├── fold1        
  │   ├── train
    │   ├── <path_to_a_folder_containing_one_tissue_class>
        │   ├── <path_to_image1.png>
        │   ├── <path_to_image2.png>
        │   ├── <...>
  │   ├── val
    │   ├── <path_to_a_folder_containing_one_tissue_class>
        │   ├── <path_to_image1.png>
        │   ├── <path_to_image2.png>
        │   ├── <...>
  │   ├── test
    │   ├── <path_to_a_folder_containing_one_tissue_class>
        │   ├── <path_to_image1.png>
        │   ├── <path_to_image2.png>
        │   ├── <...>

...(fold 2, 3)
  ├── fold4        
  │   ├── train
    │   ├── <path_to_a_folder_containing_one_tissue_class>
        │   ├── <path_to_image1.png>
        │   ├── <path_to_image2.png>
        │   ├── <...>
  │   ├── val
    │   ├── <path_to_a_folder_containing_one_tissue_class>
        │   ├── <path_to_image1.png>
        │   ├── <path_to_image2.png>
        │   ├── <...>
  │   ├── test
    │   ├── <path_to_a_folder_containing_one_tissue_class>
        │   ├── <path_to_image1.png>
        │   ├── <path_to_image2.png>
        │   ├── <...>

```

The optional input arguments are:

`num_epochs`: the maximum number of training epochs

`batches`: batch size

`learning_rate`: learning rate

- Model weights

SSL1-pretrained downstream model: `./modeling/downstream_ensemble/individual_downstream_model_weights/proposed_ssl1`

SSL2-pretrained downstream model: `./modeling/downstream_ensemble/individual_downstream_model_weights/proposed_ssl2`

SSL3-pretrained downstream model: `./modeling/downstream_ensemble/individual_downstream_model_weights/proposed_ssl3`

- Step 2: Combine the predictions of the 3 downstream models in an ensemble framework

First, you need to put the 3 downstream model weights into a folder called `individual_downstream_model_weights` under `./modeling/downstream_ensemble`. 

```
├── individual_downstream_model_weights        
  │   ├── proposed_ssl1
    │   ├── resnet18_fold0.pth
    │   ├── resnet18_fold1.pth
    │   ├── resnet18_fold2.pth
    │   ├── resnet18_fold3.pth
    │   ├── resnet18_fold4.pth

  │   ├── proposed_ssl2
    │   ├── resnet18_fold0.pth
    │   ├── resnet18_fold1.pth
    │   ├── resnet18_fold2.pth
    │   ├── resnet18_fold3.pth
    │   ├── resnet18_fold4.pth

  │   ├── proposed_ssl3
    │   ├── resnet18_fold0.pth
    │   ├── resnet18_fold1.pth
    │   ├── resnet18_fold2.pth
    │   ├── resnet18_fold3.pth
    │   ├── resnet18_fold4.pth
```

Then do:

```
cd modeling/downstream_ensemble
```

```
CUDA_VISIBLE_DEVICES=0 python main.py --path_to_all_images
```
`path_to_all_images` is the parent path that contains all folds' train, val, and test data, the same as the last step.

This script finds the best weight for each individual downstream model's prediction probability and do a weighted emsemble across the 3 models.

Finally it will print out the average F1 score on the test sets.

