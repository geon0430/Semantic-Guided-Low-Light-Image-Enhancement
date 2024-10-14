#!/bin/bash

IMAGE_NAME="low-light-enhancement-training"
TAG="0.1"

docker build --no-cache -t ${IMAGE_NAME}:${TAG} .
