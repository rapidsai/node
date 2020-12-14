ARG CUDA_VERSION=11.0
ARG NODE_VERSION=14.10.1
ARG LINUX_VERSION=ubuntu18.04
ARG CUDA_SHORT_VERSION=${CUDA_VERSION}

FROM node:$NODE_VERSION-stretch-slim as node
FROM jrottenberg/ffmpeg:4.1-nvidia AS ffmpeg

FROM nvidia/cudagl:${CUDA_VERSION}-runtime-${LINUX_VERSION}

# Install dependencies
RUN apt update -y \
 && apt install -y \
    # cuDF dependencies
    libboost-filesystem1.71.0 \
 && apt autoremove -y \
 && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ENV NODE_VERSION=$NODE_VERSION
ENV YARN_VERSION=1.22.5

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
ARG GID=1000

RUN groupadd --gid $GID node \
 && useradd --uid $UID --gid node --shell /bin/bash --create-home node \
 && ln -s /usr/local/bin/node /usr/local/bin/nodejs \
 && ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
 && ln -s /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx \
 # smoke tests
 && node --version && npm --version && yarn --version

COPY --from=ffmpeg /usr/local/bin /usr/local/bin/
COPY --from=ffmpeg /usr/local/share /usr/local/share/
COPY --from=ffmpeg /usr/local/lib /usr/local/lib/
COPY --from=ffmpeg /usr/local/include /usr/local/include/

SHELL ["/bin/bash", "-l"]

ENTRYPOINT ["docker-entrypoint.sh"]

CMD ["node"]
