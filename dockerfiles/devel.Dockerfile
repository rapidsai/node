ARG CUDA_VERSION=10.2
ARG LINUX_VERSION=ubuntu18.04
ARG CUDA_SHORT_VERSION=${CUDA_VERSION}

FROM node:14.4.0-stretch-slim as node

FROM nvidia/cudagl:${CUDA_VERSION}-devel-${LINUX_VERSION}

ENV NODE_VERSION 14.4.0
ENV YARN_VERSION 1.22.4

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
ARG PARALLEL_LEVEL=

ENV CMAKE_VERSION=3.17.2

RUN groupadd --gid $GID node \
 && useradd --uid $UID --gid node -G sudo --shell /bin/bash --create-home node \
 && echo node:node | chpasswd \
 && ln -s /usr/local/bin/node /usr/local/bin/nodejs \
 && ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
 && ln -s /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx \
 # smoke tests
 && node --version && npm --version && yarn --version \
 && sed -ri "s/32m/33m/g" /home/node/.bashrc \
 && sed -ri "s/34m/36m/g" /home/node/.bashrc \
 && mkdir -p /etc/bash_completion.d \
 && npm completion > /etc/bash_completion.d/npm \
 # Install dev dependencies and tools
 && apt update -y \
 && apt install -y software-properties-common \
 && add-apt-repository -y ppa:git-core/ppa \
 && apt install -y \
    git nano sudo wget ninja-build bash-completion \
    # ccache dependencies
    unzip automake autoconf libb2-dev libzstd-dev \
    # CMake dependencies
    curl libssl-dev libcurl4-openssl-dev zlib1g-dev \
    # X11 dependencies
    libxrandr-dev libxinerama-dev libxcursor-dev \
    # node-canvas dependencies
    libcairo2-dev libpango1.0-dev libjpeg-dev libgif-dev librsvg2-dev \
    # GLEW dependencies
    build-essential libxmu-dev libxi-dev libgl-dev libgl1-mesa-dev libglu1-mesa-dev \
 && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
 # Install CMake
 && curl -fsSLO --compressed "https://github.com/Kitware/CMake/releases/download/v$CMAKE_VERSION/cmake-$CMAKE_VERSION.tar.gz" \
 && tar -xvzf cmake-$CMAKE_VERSION.tar.gz && cd cmake-$CMAKE_VERSION \
 && ./bootstrap --system-curl --parallel=$(nproc) && make install -j$PARALLEL_LEVEL \
 && cd - && rm -rf ./cmake-$CMAKE_VERSION ./cmake-$CMAKE_VERSION.tar.gz \
 # Install ccache
 && curl -s -L https://github.com/ccache/ccache/archive/master.zip -o ccache-master.zip \
 && unzip -d ccache-master ccache-master.zip && cd ccache-master/ccache-master \
 && ./autogen.sh && ./configure --disable-man && make install -j$PARALLEL_LEVEL && cd - && rm -rf ./ccache-master*

SHELL ["/bin/bash", "-l"]

ENTRYPOINT ["docker-entrypoint.sh"]

CMD ["/bin/bash", "-l"]
