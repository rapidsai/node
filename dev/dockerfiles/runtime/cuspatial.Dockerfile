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
        /tmp/rapids/rapidsai-core-*.tgz       \
        /tmp/rapids/rapidsai-cuda-*.tgz       \
        /tmp/rapids/rapidsai-rmm-*.tgz        \
        /tmp/rapids/rapidsai-cudf-*.tgz       \
        /tmp/rapids/rapidsai-cuspatial-*.tgz; \
    for x in core cuda rmm cudf cuspatial; do \
        mkdir node_modules/@rapidsai/${x}/build/Release; \
        tar -C node_modules/@rapidsai/${x}/build/Release \
            -f /tmp/rapids/rapidsai_${x}-*-Linux.tar.gz \
            --wildcards --strip-components=2 \
            -x "**/lib/rapidsai_${x}.node" ; \
    done

FROM ${FROM_IMAGE}

SHELL ["/bin/bash", "-c"]

USER root

# Install dependencies
RUN export DEBIAN_FRONTEND=noninteractive \
 && apt update \
 && apt install -y --no-install-recommends \
    # cuSpatial dependencies
    libgdal-dev \
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
