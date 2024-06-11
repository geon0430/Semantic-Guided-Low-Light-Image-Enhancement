#!/bin/bash

port_num="5"
CONTAINER_NAME="sung-ai-video-converter"
IMAGE_NAME="hub.inbic.duckdns.org/ai-dev/ai-video-converter"
TAG="0.1"

ai_video_converter_path=$(pwd)

docker run \
    --runtime nvidia \
    --gpus all \
    -it \
    -p ${port_num}8000:8000 \
    -p ${port_num}8888:8888 \
    --name ${CONTAINER_NAME} \
    --privileged \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /raid/datasets/LOLdataset:/volume/LOLdataset \
    -v /raid/datasets/ImageNet:/volume/ImageNet \
    -v ${ai_video_converter_path}:/ai-video-converter \
    -v /home/ubuntu/geon/dataset:/volume \
    --shm-size 5g \
    --restart=always \
    -w /ai-video-converter \
    -e DISPLAY=$DISPLAY \
    ${IMAGE_NAME}:${TAG}
