
# Stage 1: Builder/Compiler
FROM python:3.10.12-slim as builder

ENV DEBIAN_FRONTEND=noninteractive
ENV VIRTUAL_ENV=/opt/venv

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        build-essential \
        gcc

WORKDIR /flow-hsi
COPY . ./

RUN python3 -m venv $VIRTUAL_ENV && \
    . $VIRTUAL_ENV/bin/activate && \
    python3 -m pip install --upgrade pip && \
    pip install .

# Stage 2: Runtime
FROM nvidia/cuda:11.6.2-runtime-ubuntu24.04

LABEL desc="flow-hsi docker container"

ENV DEBIAN_FRONTEND=noninteractive
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY --from=builder --chmod=777 $VIRTUAL_ENV $VIRTUAL_ENV

RUN apt-get update && \ 
    apt-get install --no-install-recommends -y \
        ffmpeg \
        python3 \
        python3-distutils && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3 $VIRTUAL_ENV/bin/python3

WORKDIR /workspace