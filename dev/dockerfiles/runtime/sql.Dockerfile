# syntax=docker/dockerfile:1.3

ARG FROM_IMAGE
ARG BUILD_IMAGE
ARG DEVEL_IMAGE

FROM ${BUILD_IMAGE} as build
FROM ${DEVEL_IMAGE} as devel

WORKDIR /home/node

ENV RAPIDSAI_SKIP_DOWNLOAD=1

RUN --mount=type=bind,from=build,source=/opt/rapids/,target=/tmp/rapids/ \
    npm install --omit=dev --omit=peer --omit=optional --legacy-peer-deps --force \
        /tmp/rapids/rapidsai-core-*.tgz \
        /tmp/rapids/rapidsai-cuda-*.tgz \
        /tmp/rapids/rapidsai-rmm-*.tgz  \
        /tmp/rapids/rapidsai-cudf-*.tgz \
        /tmp/rapids/rapidsai-sql-*.tgz; \
    for x in cuda rmm cudf sql; do \
        mkdir node_modules/@rapidsai/${x}/build/Release; \
        tar -C node_modules/@rapidsai/${x}/build/Release \
            -f /tmp/rapids/rapidsai_${x}-*-Linux.tar.gz \
            --wildcards --strip-components=2 \
            -x "**/lib/rapidsai_${x}.node" ; \
    done

FROM scratch as ucx-deb-amd64

ONBUILD ARG UCX_VERSION=1.12.1
ONBUILD ARG LINUX_VERSION=ubuntu20.04
ONBUILD ADD https://github.com/openucx/ucx/releases/download/v${UCX_VERSION}/ucx-v${UCX_VERSION}-${LINUX_VERSION}-mofed5-cuda11.deb /ucx.deb

FROM ucx-deb-${TARGETARCH} as ucx-deb

FROM ${FROM_IMAGE}

SHELL ["/bin/bash", "-c"]

USER root

RUN --mount=type=bind,from=ucx-deb,target=/usr/src/ucx \
 # Install dependencies
    export DEBIAN_FRONTEND=noninteractive \
 && apt update \
 && apt install -y --no-install-recommends \
    # UCX runtime dependencies
    libibverbs1 librdmacm1 libnuma1 \
    # SQL dependencies
    openjdk-8-jre-headless libboost-regex-dev libboost-system-dev libboost-filesystem-dev \
 # Install UCX
 && dpkg -i /usr/src/ucx/ucx.deb || true && apt install --fix-broken \
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
