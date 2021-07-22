ARG FROM_IMAGE
ARG DEVEL_IMAGE

FROM ${DEVEL_IMAGE} as devel

WORKDIR /home/node

RUN cp \
    /opt/node-rapids/.npmrc               \
    /opt/node-rapids/build/rapidsai-core-*.tgz  \
    /opt/node-rapids/build/nvidia-glfw-*.tgz    \
    /opt/node-rapids/build/nvidia-webgl-*.tgz   \
    . \
 && npm install --production --omit dev --omit peer --omit optional --legacy-peer-deps --force *.tgz \
 && npm dedupe  --production --omit dev --omit peer --omit optional --legacy-peer-deps --force


FROM ${FROM_IMAGE}

SHELL ["/bin/bash", "-c"]

ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics

USER root

RUN export DEBIAN_FRONTEND=noninteractive \
 && apt update --fix-missing \
 && apt install -y --no-install-recommends \
    # X11 dependencies
    libxrandr-dev libxinerama-dev libxcursor-dev \
    # GLEW dependencies
    libgl1-mesa-dev libegl1-mesa-dev libglu1-mesa-dev \
    # node-canvas dependencies
    libcairo2-dev libpango1.0-dev libjpeg-dev libgif-dev librsvg2-dev \
 # Clean up
 && apt autoremove -y && apt clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY --from=devel --chown=node:node /home/node/node_modules/ /home/node/node_modules/

USER node

WORKDIR /home/node

SHELL ["/bin/bash", "-l"]

CMD ["node"]