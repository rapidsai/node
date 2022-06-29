# syntax=docker/dockerfile:1.3

ARG FROM_IMAGE

FROM ${FROM_IMAGE} as build

WORKDIR /opt/rapids/node

ENV NVIDIA_DRIVER_CAPABILITIES all

ARG CUDAARCHS=ALL
ARG PARALLEL_LEVEL
ARG NVCC_APPEND_FLAGS
ARG RAPIDS_VERSION
ARG SCCACHE_REGION
ARG SCCACHE_BUCKET
ARG SCCACHE_IDLE_TIMEOUT

RUN echo -e "build env:\n$(env)"

COPY --chown=rapids:rapids .npmrc        /home/node/.npmrc
COPY --chown=rapids:rapids .npmrc        .npmrc
COPY --chown=rapids:rapids .yarnrc       .yarnrc
COPY --chown=rapids:rapids .eslintrc.js  .eslintrc.js
COPY --chown=rapids:rapids LICENSE       LICENSE
COPY --chown=rapids:rapids typedoc.js    typedoc.js
COPY --chown=rapids:rapids lerna.json    lerna.json
COPY --chown=rapids:rapids tsconfig.json tsconfig.json
COPY --chown=rapids:rapids package.json  package.json
COPY --chown=rapids:rapids yarn.lock     yarn.lock
COPY --chown=rapids:rapids scripts       scripts
COPY --chown=rapids:rapids modules       modules

USER root

ENV RAPIDSAI_SKIP_DOWNLOAD=1

RUN --mount=type=ssh,required=true \
    --mount=type=secret,id=sccache_credentials \
    if [ -f /run/secrets/sccache_credentials ]; then \
        export $(grep -v '^#' /run/secrets/sccache_credentials | xargs -d '\n'); \
    fi; \
    echo -e "build context:\n$(find .)" \
 && bash -c 'echo -e "\
CUDAARCHS=$CUDAARCHS\n\
PARALLEL_LEVEL=$PARALLEL_LEVEL\n\
NVCC_APPEND_FLAGS=$NVCC_APPEND_FLAGS\n\
RAPIDS_VERSION=$RAPIDS_VERSION\n\
SCCACHE_REGION=$SCCACHE_REGION\n\
SCCACHE_BUCKET=$SCCACHE_BUCKET\n\
SCCACHE_IDLE_TIMEOUT=$SCCACHE_IDLE_TIMEOUT\n\
" > .env' \
 && yarn --pure-lockfile --network-timeout 1000000 \
 && yarn build \
 && yarn dev:npm:pack \
 && chown rapids:rapids build/*.{tgz,tar.gz} \
 && mv build/*.tgz ../ && mv build/*.tar.gz ../

FROM alpine:latest

COPY --from=build /opt/rapids/*.tgz /opt/rapids/
COPY --from=build /opt/rapids/*.tar.gz /opt/rapids/
