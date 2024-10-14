#!/bin/bash

port_num="1"
CONTAINER_NAME="low-light-enhancement-training"
IMAGE_NAME="low-light-enhancement-training"
TAG="0.1"

code_path=$(pwd)

docker run \
    --runtime nvidia \
    --gpus all \
    -it \
    -p ${port_num}8888:8888 \
    --name ${CONTAINER_NAME} \
    --privileged \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    # -v /datasets/LOLdataset:/volume/LOLdataset \
    # -v /datasets/ImageNet:/volume/ImageNet \
    -v ${code_path}:/lle \
    --shm-size 5g \
    --restart=always \
    -w /lle \
    -e DISPLAY=$DISPLAY \
    ${IMAGE_NAME}:${TAG}
