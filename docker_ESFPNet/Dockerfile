FROM ubuntu:22.04
#FROM ubuntu:20.04


ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Brussels
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update
#RUN apt upgrade

RUN apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common

RUN  apt-get update \
  && apt-get install -y wget \
  && rm -rf /var/lib/apt/lists/*


RUN apt update && \
    apt install software-properties-common -y && \
    add-apt-repository ppa:graphics-drivers && \
    apt install nvidia-driver-470 -y
    #apt install nvidia-driver-510 -y

RUN apt-get update
RUN apt-get install -y software-properties-common ubuntu-drivers-common
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get install -y python3.9 python3-pip
RUN apt-get install -y wget


RUN mkdir /app
WORKDIR /app
RUN mkdir /app/inputImage
RUN mkdir /app/outputImage
RUN mkdir /app/Decoder
RUN mkdir /app/Encoder

############ Mine ######################
RUN pip3 install --upgrade pip
RUN apt-get update -y
RUN apt-get install -y libjpeg-dev zlib1g-dev
#RUN pip3 install torch==1.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
#RUN pip3 install torch --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
#RUN pip3 install torchvision==0.13.1
#RUN pip3 install torchvision --extra-index-url https://download.pytorch.org/whl/cu116
#RUN pip3 install torchaudio --extra-index-url https://download.pytorch.org/whl/cu116 
RUN pip3 install h5py
#RUN pip3 install torchnet==0.0.4
RUN pip3 install requests==2.19.1
RUN pip3 install graphviz #==2.40.1
RUN pip3 install scipy
RUN pip3 install imageio
RUN pip3 install scikit-image
RUN pip3 install timm
RUN pip3 install mmcv

#RUN pip3 install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
########################################


COPY requirements.txt /app
RUN pip3 install -r requirements.txt

RUN export CUDA_VISIBLE_DEVICES=0
RUN export NVIDIA_VISIBLE_DEVICES=all
RUN export NVIDIA_DRIVER_CAPABILITIES=all

#COPY .. /app
COPY app.py /app
COPY ESFPNet_DSM_2021_segmentation.pt /app
COPY mit_b4.pth /app
COPY mit.py /app/Encoder
COPY mlp.py /app/Decoder


EXPOSE 5000
CMD [ "python3", "app.py" ]
