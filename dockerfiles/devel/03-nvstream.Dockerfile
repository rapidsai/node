ARG FROM_IMAGE
ARG CUDA_10_2_IMAGE

FROM ${CUDA_10_2_IMAGE:-nvidia/cuda:10.2-devel-ubuntu18.04} as cuda_10_2

FROM ${FROM_IMAGE}

RUN echo -e "\
deb http://dk.archive.ubuntu.com/ubuntu/ xenial main\n\
deb http://dk.archive.ubuntu.com/ubuntu/ xenial universe\n\
" | tee /etc/apt/sources.list.d/xenial.list >/dev/null \
 && export DEBIAN_FRONTEND=noninteractive \
 && apt update --fix-missing \
 && apt install -y --no-install-recommends \
    libxi-dev libxrandr-dev unzip gcc-5 g++-5 \
 # Clean up
 && apt autoremove -y && apt clean \
 && rm -rf \
    /tmp/* \
    /var/tmp/* \
    /var/lib/apt/lists/* \
    /etc/apt/sources.list.d/xenial.list

# Copy in CUDA 10.2 headers
COPY --from=cuda_10_2 --chown=node:node /usr/local/cuda/include /opt/node-nvidia-stream-sdk/lib/cuda-10.2/include
# Copy in StreamSDK build
COPY --chown=node:node StreamSDK_Full /opt/node-nvidia-stream-sdk/StreamSDK_Full.zip
# Copy in node-nvidia-stream-sdk
COPY --chown=node:node node-nvidia-stream-sdk /opt/node-nvidia-stream-sdk

RUN chown -R node:node /opt/node-nvidia-stream-sdk

RUN cd /opt/node-nvidia-stream-sdk \
 && unzip StreamSDK_Full.zip "bin/linux/**/*" "include/**/*" -d lib/streamsdk \
 && yarn && yarn cpp:rebuild \
 && npm pack --pack-destination /home/node . \
 && chown -R node:node /home/node

WORKDIR /opt/node-rapids

USER node
