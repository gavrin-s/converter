#!/bin/bash

CONTAINER_NAME=converter

# build the dockerfile for the environment
if [ ! -z $(docker images -q "$CONTAINER_NAME:latest") ]; then
  echo "Dockerfile has already been built"
else
  echo "Building docker image"
  docker build -t "$CONTAINER_NAME" .
fi

# run the container
echo "Starting docker container"
docker run -it --rm \
  --gpus=all \
  -v `pwd`:/workspace/converter \
  "$CONTAINER_NAME"
