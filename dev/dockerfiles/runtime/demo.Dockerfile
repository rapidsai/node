# syntax=docker/dockerfile:1

ARG FROM_IMAGE
ARG BUILD_IMAGE
ARG DEVEL_IMAGE

FROM ${BUILD_IMAGE} as build
FROM ${DEVEL_IMAGE} as devel

WORKDIR /home/node

ENV RAPIDSAI_SKIP_DOWNLOAD=1

RUN --mount=type=bind,from=build,source=/opt/rapids/,target=/tmp/rapids/ \
    npm install --omit=dev --omit=peer --omit=optional --legacy-peer-deps --force \
        /tmp/rapids/rapidsai-core-*.tgz      \
        /tmp/rapids/rapidsai-cuda-*.tgz      \
        /tmp/rapids/rapidsai-glfw-*.tgz      \
        /tmp/rapids/rapidsai-webgl-*.tgz     \
        /tmp/rapids/rapidsai-rmm-*.tgz       \
        /tmp/rapids/rapidsai-cudf-*.tgz      \
        /tmp/rapids/rapidsai-cuml-*.tgz      \
        /tmp/rapids/rapidsai-cugraph-*.tgz   \
        /tmp/rapids/rapidsai-cuspatial-*.tgz \
        /tmp/rapids/rapidsai-io-*.tgz        \
        /tmp/rapids/rapidsai-deck.gl-*.tgz   \
        /tmp/rapids/rapidsai-jsdom-*.tgz     \
        /tmp/rapids/rapidsai-demo-*.tgz;     \
    for x in cuda rmm cudf cuml cugraph cuspatial io; do \
        mkdir node_modules/@rapidsai/${x}/build/Release; \
        tar -C node_modules/@rapidsai/${x}/build/Release \
            -f /tmp/rapids/rapidsai_${x}-*-Linux.tar.gz  \
            --wildcards --strip-components=1             \
            -x "**/rapidsai_${x}.node" ;                 \
    done;


FROM scratch as ucx-amd64

ONBUILD ARG CUDA_VERSION_MAJOR=12
ONBUILD ARG UCX_VERSION=1.20.0
ONBUILD ARG LINUX_VERSION=ubuntu24.04
ONBUILD ADD https://github.com/openucx/ucx/releases/download/v${UCX_VERSION}/ucx-${UCX_VERSION}-${LINUX_VERSION}-mofed5-cuda${CUDA_VERSION_MAJOR}-x86_64.tar.bz2 /ucx.tar.bz2

FROM scratch as ucx-arm64

ONBUILD ARG CUDA_VERSION_MAJOR=12
ONBUILD ARG UCX_VERSION=1.20.0
ONBUILD ARG LINUX_VERSION=ubuntu24.04
ONBUILD ADD https://github.com/openucx/ucx/releases/download/v${UCX_VERSION}/ucx-${UCX_VERSION}-${LINUX_VERSION}-mofed5-cuda${CUDA_VERSION_MAJOR}-aarch64.tar.bz2 /ucx.tar.bz2

FROM ucx-${TARGETARCH} as ucx

FROM ${FROM_IMAGE}

SHELL ["/bin/bash", "-c"]

USER root

RUN --mount=type=bind,from=ucx,target=/usr/src/ucx \
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
    # node-canvas dependencies
    libcairo2 libpango-1.0-0 libpangocairo-1.0-0 libjpeg8 libgif7 librsvg2-2 \
 # Install UCX
 && tar -C /usr/src/ucx -xvjf /usr/src/ucx/ucx.tar.bz2 \
 && apt install -y --no-install-recommends /usr/src/ucx/*.deb || true \
 && apt install -y --fix-broken \
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
