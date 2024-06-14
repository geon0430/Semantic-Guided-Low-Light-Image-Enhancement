FROM nvcr.io/nvidia/pytorch:23.01-py3

ENV TZ=Asia/Seoul
ENV DEBIAN_FRONTEND=noninteractive

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES all

ENV XDG_RUNTIME_DIR "/tmp"

WORKDIR /
RUN mkdir -p /Semantic-Guided-Low-Light-Image-Enhancement

COPY . /Semantic-Guided-Low-Light-Image-Enhancement


RUN apt-get update && \
    apt-get install -y python3 python3-pip

RUN if [ -e /usr/bin/python ]; then rm /usr/bin/python; fi && \
    ln -s $(which python3) /usr/bin/python
