# Make sure to run this with "--gpus all" for gpu access

# Also,
####################################################################################################
## This container image and its contents are governed by the NVIDIA Deep Learning Container License.
## By pulling and using the container, you accept the terms and conditions of this license:
##    https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license
####################################################################################################
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

WORKDIR /workspace

COPY . /workspace

RUN apt-get update 

# # Debugging
# RUN apt-get install -y vim 

RUN pip install -r requirements.txt

ENTRYPOINT ["python3", "main.py"]