# syntax=docker/dockerfile:1
ARG TARGETARCH

ARG FROM_IMAGE

FROM ${FROM_IMAGE} as build

ARG TARGETARCH

WORKDIR /opt/rapids/node

ENV NVIDIA_DRIVER_CAPABILITIES all

ARG CUDAARCHS=RAPIDS
ARG PARALLEL_LEVEL
ARG NVCC_APPEND_FLAGS
ARG RAPIDS_VERSION
ARG SCCACHE_REGION
ARG SCCACHE_BUCKET

RUN echo -e "build env:\n$(env)"

COPY --chown=rapids:rapids .npmrc           /home/node/.npmrc
COPY --chown=rapids:rapids .npmrc           .npmrc
COPY --chown=rapids:rapids .yarnrc          .yarnrc
COPY --chown=rapids:rapids eslint.config.js eslint.config.js
COPY --chown=rapids:rapids LICENSE          LICENSE
COPY --chown=rapids:rapids typedoc.js       typedoc.js
COPY --chown=rapids:rapids lerna.json       lerna.json
COPY --chown=rapids:rapids tsconfig.json    tsconfig.json
COPY --chown=rapids:rapids package.json     package.json
COPY --chown=rapids:rapids yarn.lock        yarn.lock
COPY --chown=rapids:rapids scripts          scripts
COPY --chown=rapids:rapids modules          modules

ENV RAPIDSAI_SKIP_DOWNLOAD=1

SHELL ["/bin/bash", "-c"]

RUN --mount=type=bind,source=dev/.gitconfig,target=/opt/rapids/.gitconfig \
    --mount=type=secret,id=aws_creds,uid=1000,gid=1000,target=/opt/rapids/.aws/credentials \
    --mount=type=secret,id=sccache_config,uid=1000,gid=1000,target=/opt/rapids/.config/sccache/config \
<<EOF_RUN

sudo chown -R $(id -u):$(id -g) /opt/rapids

echo -e "build context:\n$(find .)"

cat <<EOF > .env
CUDAARCHS=$CUDAARCHS
NO_COLOR=1
PARALLEL_LEVEL=${PARALLEL_LEVEL:-$(ulimit -n)}
CMAKE_BUILD_PARALLEL_LEVEL=${PARALLEL_LEVEL:-$(ulimit -n)}
NVCC_APPEND_FLAGS=$NVCC_APPEND_FLAGS
RAPIDS_VERSION=$RAPIDS_VERSION
SCCACHE_SERVER_LOG=sccache=debug
SCCACHE_ERROR_LOG=/tmp/sccache.log
SCCACHE_REGION=$SCCACHE_REGION
SCCACHE_BUCKET=$SCCACHE_BUCKET
SCCACHE_DIST_AUTH_TYPE=token
SCCACHE_DIST_MAX_RETRIES=inf
SCCACHE_DIST_REQUEST_TIMEOUT=7140
SCCACHE_DIST_FALLBACK_TO_LOCAL_COMPILE=false
SCCACHE_DIST_SCHEDULER_URL=https://${TARGETARCH}.linux.sccache.rapids.nvidia.com
SCCACHE_S3_KEY_PREFIX=/node
SCCACHE_S3_PREPROCESSOR_CACHE_KEY_PREFIX=/node/preprocessor
SCCACHE_S3_USE_PREPROCESSOR_CACHE_MODE=true
EOF

yarn --pure-lockfile --network-timeout 1000000

yarn build

yarn dev:npm:pack

chown rapids:rapids build/*.{tgz,tar.gz}

mv build/*.{tgz,tar.gz} ../

EOF_RUN


FROM alpine:latest

COPY --from=build /opt/rapids/*.tgz /opt/rapids/
COPY --from=build /opt/rapids/*.tar.gz /opt/rapids/
