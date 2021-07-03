ARG BASE_IMAGE
ARG BUILD_IMAGE
ARG NODE_VERSION=15.14.0

FROM node:$NODE_VERSION-stretch-slim as node

FROM ${BUILD_IMAGE} as build

COPY --chown=node:node build/*.tgz /home/node/

SHELL ["/bin/bash", "-c"]

RUN cd /home/node \
 && npm install \
    --no-fund --no-audit \
    --production --save-exact \
    --omit dev --omit peer --omit optional \
    /home/node/*.tgz \
 && npm dedupe \
    --no-fund --no-audit \
    --production --save-exact \
    --omit dev --omit peer --omit optional

FROM ${BASE_IMAGE}

# Install dependencies
RUN export DEBIAN_FRONTEND=noninteractive \
 && apt update -y \
 && apt install --no-install-recommends -y \
    # X11 dependencies
    libxrandr-dev libxinerama-dev libxcursor-dev \
    # GLEW dependencies
    libgl1-mesa-dev libegl1-mesa-dev libglu1-mesa-dev \
    # node-canvas dependencies
    libcairo2-dev libpango1.0-dev libjpeg-dev libgif-dev librsvg2-dev \
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

COPY --from=build --chown=node:node /home/node/node_modules /home/node/node_modules

WORKDIR /home/node

CMD ["node"]
