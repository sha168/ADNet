FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel
RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get -y install curl
RUN curl https://bootstrap.pypa.io/get-pip.py | python3
RUN pip3 install --no-cache matplotlib
RUN pip3 install scipy
RUN pip3 install scikit-learn
RUN pip3 install json5==0.8.5
RUN pip3 install opencv-python 
RUN pip3 install Pillow==7.1.0
RUN pip3 install scikit-image==0.14.2
RUN pip3 install SimpleITK==1.2.3
RUN pip3 install nibabel
RUN pip3 install pymongo
RUN pip3 install sklearn
RUN pip3 install --no-cache PyGithub
RUN pip3 install monai
