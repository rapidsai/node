# syntax=docker/dockerfile:1

ARG BASE_IMAGE
ARG DEVEL_IMAGE
ARG NODE_VERSION=15.14.0

FROM node:$NODE_VERSION-stretch-slim as node
FROM nvidia/cuda:10.2-devel-ubuntu18.04 as cuda102

FROM ${DEVEL_IMAGE} as devel

ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES},video

SHELL ["/bin/bash", "-c"]

USER root

RUN echo -e "\
deb http://dk.archive.ubuntu.com/ubuntu/ xenial main\n\
deb http://dk.archive.ubuntu.com/ubuntu/ xenial universe\n\
" | tee /etc/apt/sources.list.d/xenial.list >/dev/null \
 && apt update --fix-missing \
 # Install unzip, ssh client and gcc-5
 && apt install -y --no-install-recommends unzip openssh-client gcc-5 g++-5 \
 # Download public key for github.com
 && mkdir -p -m 0600 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts \
 # Clean up
 && apt autoremove -y && apt clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Clone the private node-nvidia-stream-sdk
RUN --mount=type=ssh git clone git@github.com:trxcllnt/node-nvidia-stream-sdk.git /opt/node-nvidia-stream-sdk \
 && chown -R node:node /opt/node-nvidia-stream-sdk

USER node

WORKDIR /opt/node-nvidia-stream-sdk

# Copy in StreamSDK build
COPY --chown=node:node StreamSDK_Full /opt/node-nvidia-stream-sdk/StreamSDK_Full.zip
# Copy in CUDA 10.2 headers
COPY --from=cuda102 --chown=node:node /usr/local/cuda/include /opt/node-nvidia-stream-sdk/lib/cuda-10.2/include

ARG PARALLEL_LEVEL
ARG RAPIDS_VERSION
ARG SCCACHE_REGION
ARG SCCACHE_BUCKET
ARG SCCACHE_CACHE_SIZE
ARG SCCACHE_IDLE_TIMEOUT
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY

RUN echo -e "build env:\n$(env)" \
 && echo -e "build context:\n$(find .)" \
 && export PARALLEL_LEVEL="$PARALLEL_LEVEL" \
 && export RAPIDS_VERSION="$RAPIDS_VERSION" \
 && export SCCACHE_REGION="$SCCACHE_REGION" \
 && export SCCACHE_BUCKET="$SCCACHE_BUCKET" \
 && export SCCACHE_CACHE_SIZE="$SCCACHE_CACHE_SIZE" \
 && export SCCACHE_IDLE_TIMEOUT="$SCCACHE_IDLE_TIMEOUT" \
 && export AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
 && export AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
 && unzip StreamSDK_Full.zip "bin/linux/**/*" "include/**/*" -d lib/streamsdk \
 && yarn \
 && yarn cpp:rebuild \
 && npm pack --pack-destination /home/node .

WORKDIR /home/node

RUN npm install --production --omit dev --omit peer --omit optional --legacy-peer-deps --force *.tgz \
 && npm dedupe  --production --omit dev --omit peer --omit optional --legacy-peer-deps --force \
 && rm *.tgz

FROM ${BASE_IMAGE}

ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES},video

SHELL ["/bin/bash", "-c"]

WORKDIR /home/node

COPY --from=devel --chown=node:node \
    /home/node/node_modules/node-nvidia-stream-sdk \
    /home/node/node_modules/node-nvidia-stream-sdk

SHELL ["/bin/bash", "-l"]

CMD ["node"]
