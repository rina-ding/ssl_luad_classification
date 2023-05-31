FROM nvcr.io/nvidia/pytorch:21.12-py3

ARG USER_ID
ARG GROUP_ID
RUN pip install segmentation_models_pytorch
RUN pip install natsort
RUN pip install scikit-image
RUN pip install lightly
RUN pip install staintools
RUN pip install spams
RUN apt-get update -y
RUN apt-get install -y openslide-tools
RUN apt-get install python3-openslide
RUN pip install openslide-python
RUN pip install "opencv-python-headless<4.3"

RUN if [ ${USER_ID:-0} -ne 0 ] && [ ${GROUP_ID:-0} -ne 0 ]; then \
	groupadd -g ${GROUP_ID} requirements && \
	useradd -l -u ${USER_ID} -g requirements requirements && \
	install -d -m 0755 -o docker-user -g requirements ./requirements \
	;fi

USER requirements  
