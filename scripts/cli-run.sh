#!/bin/bash

CONTAINER_NAME=flow-hsi
IMAGE_NAME=$(id -u -n)/flow-hsi:latest

WORK_PATH=~/work/flow-hsi
DATA_PATH=/data/korepanov/datasets

#-u $(id -u):$(id -g) \
docker run --rm -it --shm-size=24g \
    --runtime=nvidia --gpus all \
    -p 8888:8888 -p 6006:6006 \
    -v $WORK_PATH:/flow-hsi \
    -v $DATA_PATH:/data \
    --name $CONTAINER_NAME \
    $IMAGE_NAME