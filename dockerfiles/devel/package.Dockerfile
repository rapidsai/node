# syntax=docker/dockerfile:1.3

ARG FROM_IMAGE

FROM ${FROM_IMAGE}

WORKDIR /opt/rapids/node

ENV NVIDIA_DRIVER_CAPABILITIES all

ARG CUDAARCHS=ALL
ARG PARALLEL_LEVEL
ARG RAPIDS_VERSION
ARG SCCACHE_REGION
ARG SCCACHE_BUCKET
ARG SCCACHE_IDLE_TIMEOUT

RUN echo -e "build env:\n$(env)"

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY

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

RUN --mount=type=secret,id=AWS_ACCESS_KEY_ID \
    --mount=type=secret,id=AWS_SECRET_ACCESS_KEY \
    export AWS_ACCESS_KEY_ID="$(cat /run/secrets/AWS_ACCESS_KEY_ID 2>/dev/null || echo $AWS_ACCESS_KEY_ID)"; \
    export AWS_SECRET_ACCESS_KEY="$(cat /run/secrets/AWS_SECRET_ACCESS_KEY 2>/dev/null || echo $AWS_SECRET_ACCESS_KEY)"; \
    echo -e "build context:\n$(find .)" \
 && bash -c 'echo -e "\
CUDAARCHS=$CUDAARCHS\n\
PARALLEL_LEVEL=$PARALLEL_LEVEL\n\
RAPIDS_VERSION=$RAPIDS_VERSION\n\
SCCACHE_REGION=$SCCACHE_REGION\n\
SCCACHE_BUCKET=$SCCACHE_BUCKET\n\
SCCACHE_IDLE_TIMEOUT=$SCCACHE_IDLE_TIMEOUT\n\
" > .env' \
 && yarn --pure-lockfile \
 && yarn tsc:build \
 && yarn cpp:build \
 && yarn dev:npm:pack \
 && yarn cache clean \
 && cp build/*.tgz ../ \
 && cd .. && rm -rf node

WORKDIR /home/node
