FROM tensorflow/tensorflow:latest-gpu 

RUN apt-get update && \ 
    apt-get install -y openssh-server rsync ffmpeg libsm6 libxext6 

RUN apt-get install -y libfontconfig1 libxrender1 

RUN pip install -U  scikit-video scikit-image sk-video opencv-python moviepy requests 

RUN apt-get install -y bc 

RUN apt-get update && apt-get install -y libhdf5-dev

RUN apt-get update  && pip install -U h5py

RUN pip install jupyterlab

RUN curl -sL https://deb.nodesource.com/setup_8.x | bash - && \
    apt-get install -y nodejs

RUN mkdir /jupyterlab_p/

WORKDIR /jupyterlab_p/

RUN apt-get install -y git-core

RUN git clone https://github.com/jupyterlab/jupyterlab-monaco.git monaco

WORKDIR /jupyterlab_p/monaco

RUN jupyter labextension install @jupyterlab/git

RUN pip install jupyterlab-git && jupyter serverextension enable --py jupyterlab_git

RUN jupyter labextension install jupyterlab-drawio


RUN pip install ipympl
# If using JupyterLab
# Install nodejs: https://nodejs.org/en/download/
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager
RUN jupyter labextension install jupyter-matplotlib
RUN pip install jupyter-tensorboard && jupyter labextension install jupyterlab_tensorboard
RUN jupyter labextension install jupyterlab-xyz-extension

WORKDIR /notebooks/