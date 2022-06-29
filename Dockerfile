FROM nvidia/cuda:11.1-devel-ubuntu18.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

# Install lib. for opencv
RUN apt-get update -y && \
    apt-get -y install libgl1-mesa-glx && \
    apt-get -y install libglib2.0-0 && \
    apt-get -y install libgtk2.0-dev

# Install Dependencies of Miniconda
RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda3
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

# Setting Work Directory
WORKDIR /ssp

# Conda Environment 
ARG conda_env=NIA21

COPY environment.yaml       /ssp
RUN conda env create -f environment.yaml

ENV PATH /opt/conda/envs/$conda_env/bin:$PATH
ENV CONDA_DEFAULT_ENV $conda_env

# Deep Learning
COPY *.py                   /ssp/

CMD python valid.py \
        --datacfg data/ape/ape.data \
        --modelcfg cfg/yolo-pose.cfg \
        --weightfile data/ape/model/model_backup.weights

