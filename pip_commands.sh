apt-get update -y
apt-get install -y openslide-tools
apt-get install python3-openslide -y
pip install openslide-python 
pip install scikit-image
pip install natsort
pip install segmentation_models_pytorch
cd preprocessing
cd spams-python
pip install -e .