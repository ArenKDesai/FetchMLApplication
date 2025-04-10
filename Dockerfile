# Make sure to run this with "--gpus all" for gpu access
# and, optionally, "--ipc=host" to speed up multi-processing

# Also,
####################################################################################################
## This container image and its contents are governed by the NVIDIA Deep Learning Container License.
## By pulling and using the container, you accept the terms and conditions of this license:
##    https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license
####################################################################################################
FROM nvidia-container-runtime

WORKDIR /workspace

COPY . /workspace

RUN pip install -r requirements.txt

# NVIDIA Container Toolkit
# RUN apt-get install curl
# RUN curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
#         && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
#         sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
#         sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
# RUN apt-get update
# RUN apt-get install -y nvidia-container-toolkit

# ENTRYPOINT ["python3", "main.py"]