FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-devel
ADD . /
WORKDIR /

RUN apt-get update
RUN apt-get install -y ffmpeg libglib2.0-0 libsm6 libxrender1 libxext6 libgl1-mesa-glx libgl1-mesa-dev rsync
RUN pip install multipledispatch
RUN pip install seaborn
RUN pip install ray ray[rllib] ray[tune]
RUN pip install pandas tabulate gym dm-tree
RUN pip install numpy scipy matplotlib scikit-learn jupyter networkx
RUN pip install opencv-contrib-python
RUN pip install tensorflow
RUN pip install lz4
RUN pip install tensorboardX gputil
RUN pip install cvxopt cvxpy
RUN cp space_utils.py /opt/conda/lib/python3.7/site-packages/ray/rllib/utils/spaces/