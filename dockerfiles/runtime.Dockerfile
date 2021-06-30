ARG BASE_IMAGE
ARG BUILD_IMAGE
ARG NODE_VERSION=15.14.0

FROM node:$NODE_VERSION-stretch-slim as node

FROM ${BUILD_IMAGE} as build

COPY --chown=node:node . /opt/rapids

ARG SCCACHE_REGION=us-west-2
ENV SCCACHE_REGION="$SCCACHE_REGION"
ARG SCCACHE_BUCKET=node-rapids-sccache
ENV SCCACHE_BUCKET="$SCCACHE_BUCKET"
ARG SCCACHE_IDLE_TIMEOUT=32768
ENV SCCACHE_IDLE_TIMEOUT="$SCCACHE_IDLE_TIMEOUT"
ARG AWS_ACCESS_KEY_ID
ENV AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID"
ARG AWS_SECRET_ACCESS_KEY
ENV AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY"

ARG DISPLAY=
ENV DISPLAY="$DISPLAY"
ARG CUDAARCHS=ALL
ENV CUDAARCHS="$CUDAARCHS"
ARG CMAKE_CUDA_ARCHITECTURES=NATIVE
ENV CMAKE_CUDA_ARCHITECTURES="$CMAKE_CUDA_ARCHITECTURES"
ARG NODE_RAPIDS_USE_LOCAL_DEPS_BUILD_DIRS=YES
ENV NODE_RAPIDS_USE_LOCAL_DEPS_BUILD_DIRS="$NODE_RAPIDS_USE_LOCAL_DEPS_BUILD_DIRS"

RUN cd /opt/rapids \
    && yarn nuke:from:orbit \
    && NODE_RAPIDS_MODULE_PATHS=$(yarn lerna exec --scope "@nvidia/*" --scope "@rapidsai/*" "echo \$PWD") \
 && cd /home/node \
    && npm pack ${NODE_RAPIDS_MODULE_PATHS} \
    && npm install --no-audit --no-fund /home/node/{nvidia,rapidsai}-*.tgz \
    && rm -rf /home/node/{nvidia,rapidsai}-*.tgz

FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt update -y \
 && apt install --no-install-recommends -y \
    # cuSpatial dependencies
    libgdal-dev \
 && apt autoremove -y \
 && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ARG NODE_VERSION
ENV NODE_VERSION=$NODE_VERSION

ARG YARN_VERSION=1.22.5
ENV YARN_VERSION=$YARN_VERSION

# Install node
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/include/node /usr/local/include/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
# Install yarn
COPY --from=node /opt/yarn-v$YARN_VERSION/bin/yarn /usr/local/bin/yarn
COPY --from=node /opt/yarn-v$YARN_VERSION/bin/yarn.js /usr/local/bin/yarn.js
COPY --from=node /opt/yarn-v$YARN_VERSION/bin/yarn.cmd /usr/local/bin/yarn.cmd
COPY --from=node /opt/yarn-v$YARN_VERSION/bin/yarnpkg /usr/local/bin/yarnpkg
COPY --from=node /opt/yarn-v$YARN_VERSION/bin/yarnpkg.cmd /usr/local/bin/yarnpkg.cmd
COPY --from=node /opt/yarn-v$YARN_VERSION/lib/cli.js /usr/local/lib/cli.js
COPY --from=node /opt/yarn-v$YARN_VERSION/lib/v8-compile-cache.js /usr/local/lib/v8-compile-cache.js
# Copy entrypoint
COPY --from=node /usr/local/bin/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

ARG UID=1000
ARG ADDITIONAL_GROUPS=

RUN useradd --uid $UID --user-group ${ADDITIONAL_GROUPS} --shell /bin/bash --create-home node \
 && ln -s /usr/local/bin/node /usr/local/bin/nodejs \
 && ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
 && ln -s /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx \
 # smoke tests
 && node --version && npm --version && yarn --version

# avoid "OSError: library nvvm not found" error
ENV CUDA_HOME="/usr/local/cuda"

SHELL ["/bin/bash", "-l"]

ENTRYPOINT ["docker-entrypoint.sh"]

USER node

WORKDIR /home/node

ENV NODE_PATH=/home/node/node_modules

COPY --from=build --chown=node:node /home/node/node_modules .

CMD ["node"]
