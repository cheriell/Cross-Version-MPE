FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime
# This also works (tested):
# FROM pytorch/pytorch:2.2.1-cuda11.8-cudnn8-runtime

RUN apt update
RUN apt-get update
RUN apt install -y vim

COPY ./requirements.txt /home/requirements.txt

RUN conda install -y pip
RUN pip install --upgrade pip
RUN pip install -r /home/requirements.txt

RUN python -c "import torch; print(torch.__version__)"
RUN python -c "import librosa; print(librosa.__version__)"
RUN python -c "import mir_eval; print(mir_eval.__version__)"
RUN python -c "import json"
