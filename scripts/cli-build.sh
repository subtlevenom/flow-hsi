#!/bin/bash

CONTAINER_NAME=flow-hsi

docker build --tag=$(id -u -n)/$CONTAINER_NAME:latest --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) ../