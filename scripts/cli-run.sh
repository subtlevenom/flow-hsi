#!/bin/bash

IMAGE_NAME=$(id -u -n)/flow-hsi:latest
CONTAINER_NAME=flow-hsi

WORK_PATH=~/work/flow-hsi
DATA_PATH=~/data

#-u $(id -u):$(id -g) \
docker run --rm -it \
    -p 8888:8888 -p 6006:6006 \
    -v $WORK_PATH:/flow-hsi \
    -v $DATA_PATH:/data \
    --name $CONTAINER_NAME \
    $IMAGE_NAME