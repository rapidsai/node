#syntax=docker/dockerfile:1.2

ARG FROM_IMAGE

FROM ${FROM_IMAGE}

ENV NVIDIA_DRIVER_CAPABILITIES all

COPY --chown=node:node .eslintrc.js       .eslintrc.js
COPY --chown=node:node .yarnrc            .yarnrc
COPY --chown=node:node LICENSE            LICENSE
COPY --chown=node:node lerna.json         lerna.json
COPY --chown=node:node .npmrc             .npmrc
COPY --chown=node:node tsconfig.json      tsconfig.json
COPY --chown=node:node typedoc.js         typedoc.js
COPY --chown=node:node package.json       package.json
COPY --chown=node:node scripts            scripts
COPY --chown=node:node modules/blazingsql modules/blazingsql
COPY --chown=node:node modules/core       modules/core
COPY --chown=node:node modules/cuda       modules/cuda
COPY --chown=node:node modules/cudf       modules/cudf
COPY --chown=node:node modules/cugraph    modules/cugraph
COPY --chown=node:node modules/cuml       modules/cuml
COPY --chown=node:node modules/cuspatial  modules/cuspatial
COPY --chown=node:node modules/deck.gl    modules/deck.gl
COPY --chown=node:node modules/glfw       modules/glfw
COPY --chown=node:node modules/jsdom      modules/jsdom
COPY --chown=node:node modules/rmm        modules/rmm
COPY --chown=node:node modules/webgl      modules/webgl

ARG CUDAARCHS=ALL
ARG PARALLEL_LEVEL
ARG RAPIDS_VERSION
ARG SCCACHE_REGION
ARG SCCACHE_BUCKET
ARG SCCACHE_CACHE_SIZE
ARG SCCACHE_IDLE_TIMEOUT

RUN echo -e "build env:\n$(env)" \
 && echo -e "build context:\n$(find .)" \
 && touch .env && yarn

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY

RUN --mount=type=secret,id=aws_creds \
 [[ -f /run/secrets/aws_creds ]] && \
    set -a && . /run/secrets/aws_creds && set +a \
    || true \
 && yarn rebuild

# Copy demos after building so changes to demos don't trigger a rebuild
COPY --chown=node:node modules/demo modules/demo
COPY --chown=node:node .npmrc       /home/node/.npmrc

RUN yarn \
 && yarn dev:npm:pack \
 && ls -all /opt/rapids/node/build
