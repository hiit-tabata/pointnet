FROM tensorflow/tensorflow:latest-gpu 

RUN apt-get update && \ 
    apt-get install -y openssh-server rsync ffmpeg libsm6 libxext6 

RUN apt-get install -y libfontconfig1 libxrender1 

RUN pip install -U  scikit-video scikit-image sk-video opencv-python moviepy requests 

RUN apt-get install -y bc 

RUN apt-get update && apt-get install -y libhdf5-dev

RUN pip install -U h5py

RUN pip install jupyterlab