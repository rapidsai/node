# syntax=docker/dockerfile:1.3

ARG FROM_IMAGE
ARG BUILD_IMAGE
ARG DEVEL_IMAGE

FROM ${BUILD_IMAGE} as build
FROM ${DEVEL_IMAGE} as devel

WORKDIR /home/node

RUN --mount=type=bind,from=build,source=/opt/rapids/,target=/tmp/rapids/ \
    npm install --omit=dev --omit=peer --omit=optional --legacy-peer-deps --force \
       /tmp/rapids/wrtc-dev.tgz             \
       /tmp/rapids/rapidsai-core-*.tgz      \
       /tmp/rapids/rapidsai-cuda-*.tgz      \
       /tmp/rapids/rapidsai-glfw-*.tgz      \
       /tmp/rapids/rapidsai-webgl-*.tgz     \
       /tmp/rapids/rapidsai-rmm-*.tgz       \
       /tmp/rapids/rapidsai-cudf-*.tgz      \
       /tmp/rapids/rapidsai-sql-*.tgz       \
       /tmp/rapids/rapidsai-cuml-*.tgz      \
       /tmp/rapids/rapidsai-cugraph-*.tgz   \
       /tmp/rapids/rapidsai-cuspatial-*.tgz \
       /tmp/rapids/rapidsai-io-*.tgz        \
       /tmp/rapids/rapidsai-deck.gl-*.tgz   \
       /tmp/rapids/rapidsai-jsdom-*.tgz     \
       /tmp/rapids/rapidsai-demo-*.tgz      ;

FROM scratch as ucx-deb-amd64

ONBUILD ARG UCX_VERSION=1.12.1
ONBUILD ARG LINUX_VERSION=ubuntu20.04
ONBUILD ADD https://github.com/openucx/ucx/releases/download/v${UCX_VERSION}/ucx-v${UCX_VERSION}-${LINUX_VERSION}-mofed5-cuda11.deb /ucx.deb

FROM ucx-deb-${TARGETARCH} as ucx-deb

FROM ${FROM_IMAGE}

SHELL ["/bin/bash", "-c"]

USER root

RUN --mount=type=bind,from=ucx-deb,target=/tmp/ucx \
 # Install dependencies
    export DEBIAN_FRONTEND=noninteractive \
 && apt update \
 && apt install -y --no-install-recommends \
    # cuSpatial dependencies
    libgdal-dev \
    # X11 dependencies
    libxrandr2 libxinerama1 libxcursor1 \
    # Wayland dependencies
    wayland-protocols \
    libwayland-{bin,egl1,cursor0,client0,server0} \
    libxkbcommon0 libxkbcommon-x11-0 \
    # GLEW dependencies
    libglvnd0 libgl1 libglx0 libegl1 libgles2 libglu1-mesa \
    # UCX runtime dependencies
    libibverbs1 librdmacm1 libnuma1 \
    # node-canvas dependencies
    libcairo2 libpango-1.0-0 libpangocairo-1.0-0 libjpeg8 libgif7 librsvg2-2 \
    # SQL dependencies
    openjdk-8-jre-headless libboost-regex-dev libboost-system-dev libboost-filesystem-dev \
 # Install UCX
 && dpkg -i /tmp/ucx/ucx.deb || true && apt install --fix-broken \
 # Clean up
 && apt autoremove -y && apt clean \
 && rm -rf \
    /tmp/* \
    /var/tmp/* \
    /var/lib/apt/lists/* \
    /var/cache/apt/archives/*

USER node

WORKDIR /home/node

COPY --from=devel --chown=node:node /home/node/node_modules node_modules

SHELL ["/bin/bash", "-l"]

CMD ["node"]
