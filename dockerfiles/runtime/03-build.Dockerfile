ARG FROM_IMAGE

FROM ${FROM_IMAGE}

ENV NVIDIA_DRIVER_CAPABILITIES all

COPY --chown=node:node scripts       scripts
COPY --chown=node:node modules       modules
COPY --chown=node:node .eslintrc.js  .eslintrc.js
COPY --chown=node:node .yarnrc       .yarnrc
COPY --chown=node:node LICENSE       LICENSE
COPY --chown=node:node lerna.json    lerna.json
COPY --chown=node:node .npmrc        .npmrc
COPY --chown=node:node tsconfig.json tsconfig.json
COPY --chown=node:node typedoc.js    typedoc.js
COPY --chown=node:node package.json  package.json

ARG DISPLAY
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

RUN env DISPLAY="$DISPLAY" \
        CUDAARCHS="$CUDAARCHS" \
        PARALLEL_LEVEL="$PARALLEL_LEVEL" \
        RAPIDS_VERSION="$RAPIDS_VERSION" \
        SCCACHE_REGION="$SCCACHE_REGION" \
        SCCACHE_BUCKET="$SCCACHE_BUCKET" \
        SCCACHE_CACHE_SIZE="$SCCACHE_CACHE_SIZE" \
        SCCACHE_IDLE_TIMEOUT="$SCCACHE_IDLE_TIMEOUT" \
        AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
        AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
    yarn rebuild

RUN yarn dev:npm:pack \
 && ls -all /opt/node-rapids/build
