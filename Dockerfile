
# Stage 1: Builder/Compiler
FROM python:3.10.12-slim AS builder

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
    pip install -r ./requirements.txt

# Stage 2: Runtime
FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04

LABEL desc="flow-hsi docker container"

ENV DEBIAN_FRONTEND=noninteractive
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY --from=builder --chmod=777 $VIRTUAL_ENV $VIRTUAL_ENV

RUN apt-get update && \
    apt-get install --no-install-recommends -y python3 && \
    apt-get install --no-install-recommends -y vifm && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3 $VIRTUAL_ENV/bin/python3 && \
    . $VIRTUAL_ENV/bin/activate

WORKDIR /flow-hsi