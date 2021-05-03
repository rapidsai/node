ARG BASE_IMAGE
ARG NODE_VERSION=15.14.0

FROM node:$NODE_VERSION-stretch-slim as node

FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

# Install dev dependencies and tools
RUN GCC_VERSION=$(bash -c '\
CUDA_VERSION=$(nvcc --version | head -n4 | tail -n1 | cut -d" " -f5 | cut -d"," -f1); \
CUDA_VERSION_MAJOR=$(echo $CUDA_VERSION | tr -d '.' | cut -c 1-2); \
CUDA_VERSION_MINOR=$(echo $CUDA_VERSION | tr -d '.' | cut -c 3); \
  if [[ "$CUDA_VERSION_MAJOR" == 9 ]]; then echo "7"; \
elif [[ "$CUDA_VERSION_MAJOR" == 10 ]]; then echo "8"; \
elif [[ "$CUDA_VERSION_MAJOR" == 11 ]]; then echo "9"; \
else echo "10"; \
fi') \
 && apt update -y \
 && apt install --no-install-recommends -y wget software-properties-common \
 && add-apt-repository --no-update -y ppa:git-core/ppa \
 && add-apt-repository --no-update -y ppa:ubuntu-toolchain-r/test \
 # Install LLVM apt sources
 && wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - \
 && touch /etc/apt/sources.list.d/llvm.list \
 && echo "deb http://apt.llvm.org/$(lsb_release -cs)/ llvm-toolchain-$(lsb_release -cs) main" >> /etc/apt/sources.list.d/llvm.list \
 && echo "deb-src  http://apt.llvm.org/$(lsb_release -cs)/ llvm-toolchain-$(lsb_release -cs) main" >> /etc/apt/sources.list.d/llvm.list \
 && echo "deb http://apt.llvm.org/$(lsb_release -cs)/ llvm-toolchain-$(lsb_release -cs)-12 main" >> /etc/apt/sources.list.d/llvm.list \
 && echo "deb-src  http://apt.llvm.org/$(lsb_release -cs)/ llvm-toolchain-$(lsb_release -cs)-12 main" >> /etc/apt/sources.list.d/llvm.list \
 && apt update -y \
 && apt install --no-install-recommends -y \
    gcc-${GCC_VERSION} g++-${GCC_VERSION} \
    jq git entr nano sudo ninja-build bash-completion \
    # Install gdb, lldb (for llnode), and clangd for C++ intellisense and debugging in the container
    gdb lldb clangd clang-format-12 \
    # ccache dependencies
    unzip automake autoconf libb2-dev libzstd-dev \
    # CMake dependencies
    curl libssl-dev libcurl4-openssl-dev zlib1g-dev \
    # X11 dependencies
    libxrandr-dev libxinerama-dev libxcursor-dev \
    # node-canvas dependencies
    libcairo2-dev libpango1.0-dev libjpeg-dev libgif-dev librsvg2-dev \
    # GLEW dependencies
    build-essential libxmu-dev libxi-dev libgl1-mesa-dev libegl1-mesa-dev libglu1-mesa-dev \
    # cuDF dependencies
    libboost-filesystem-dev \
    # cuSpatial dependencies
    libgdal-dev \
 && apt autoremove -y \
 && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
 # Remove any existing gcc and g++ alternatives
 && update-alternatives --remove-all cc  >/dev/null 2>&1 || true \
 && update-alternatives --remove-all c++ >/dev/null 2>&1 || true \
 && update-alternatives --remove-all gcc >/dev/null 2>&1 || true \
 && update-alternatives --remove-all g++ >/dev/null 2>&1 || true \
 && update-alternatives --remove-all gcov >/dev/null 2>&1 || true \
 && update-alternatives \
    --install /usr/bin/gcc gcc /usr/bin/gcc-${GCC_VERSION} 100 \
    --slave /usr/bin/cc cc /usr/bin/gcc-${GCC_VERSION} \
    --slave /usr/bin/g++ g++ /usr/bin/g++-${GCC_VERSION} \
    --slave /usr/bin/c++ c++ /usr/bin/g++-${GCC_VERSION} \
    --slave /usr/bin/gcov gcov /usr/bin/gcov-${GCC_VERSION} \
 # Set gcc-${GCC_VERSION} as the default gcc
 && update-alternatives --set gcc /usr/bin/gcc-${GCC_VERSION} \
 # Set alternative for llvm-config so it's in the path for llnode
 && LLMV_VERSION=$(lldb --version | cut -d" " -f3 | cut -d"." -f1) \
 && update-alternatives --remove-all llvm-config >/dev/null 2>&1 || true \
 && update-alternatives --install /usr/bin/llvm-config llvm-config /usr/bin/llvm-config-${LLMV_VERSION} 100 \
 && update-alternatives --set llvm-config /usr/bin/llvm-config-${LLMV_VERSION}

ARG PARALLEL_LEVEL=4
ARG CMAKE_VERSION=3.20.2

# Install CMake
RUN cd /tmp \
 && curl -fsSLO --compressed "https://github.com/Kitware/CMake/releases/download/v$CMAKE_VERSION/cmake-$CMAKE_VERSION.tar.gz" -o /tmp/cmake-$CMAKE_VERSION.tar.gz \
 && tar -xvzf /tmp/cmake-$CMAKE_VERSION.tar.gz && cd /tmp/cmake-$CMAKE_VERSION \
 && /tmp/cmake-$CMAKE_VERSION/bootstrap \
    --system-curl \
    --parallel=$PARALLEL_LEVEL \
 && make install -j$PARALLEL_LEVEL \
 && cd /tmp && rm -rf /tmp/cmake-$CMAKE_VERSION*

ARG CCACHE_VERSION=4.1

 # Install ccache
RUN cd /tmp \
 && curl -fsSLO --compressed https://github.com/ccache/ccache/releases/download/v$CCACHE_VERSION/ccache-$CCACHE_VERSION.tar.gz -o /tmp/ccache-$CCACHE_VERSION.tar.gz \
 && tar -xvzf /tmp/ccache-$CCACHE_VERSION.tar.gz && cd /tmp/ccache-$CCACHE_VERSION \
 && mkdir -p /tmp/ccache-$CCACHE_VERSION/build \
 && cd /tmp/ccache-$CCACHE_VERSION/build \
 && cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DZSTD_FROM_INTERNET=ON \
    -DENABLE_TESTING=OFF \
    /tmp/ccache-$CCACHE_VERSION \
 && make install -j$PARALLEL_LEVEL \
 && cd /tmp && rm -rf /tmp/ccache-$CCACHE_VERSION*

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
 && echo node:node | chpasswd \
 && ln -s /usr/local/bin/node /usr/local/bin/nodejs \
 && ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
 && ln -s /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx \
 # smoke tests
 && node --version && npm --version && yarn --version \
 && sed -ri "s/32m/33m/g" /home/node/.bashrc \
 && sed -ri "s/34m/36m/g" /home/node/.bashrc \
 # persist infinite bash history on the host
 && bash -c 'echo -e "\
# Infinite bash history\n\
export HISTSIZE=-1;\n\
export HISTFILESIZE=-1;\n\
export HISTCONTROL=ignoreboth;\n\
# flush commands to .bash_history immediately\n\
export PROMPT_COMMAND=\"history -a; \$PROMPT_COMMAND\";\n\
# Change the file location because certain bash sessions truncate .bash_history file upon close.\n\
# http://superuser.com/questions/575479/bash-history-truncated-to-500-lines-on-each-login\n\
export HISTFILE=\"\$DOCKER_WORKDIR/modules/.cache/.eternal_bash_history\";\n\
"' >> /home/node/.bashrc \
 # Modify the entrypoint script to export the entrypoint as a DOCKER_WORKDIR env var
 && sed -ri 's/exec "\$@"/export DOCKER_WORKDIR="\$(pwd)";\nexec "\$@"/g' /usr/local/bin/docker-entrypoint.sh \
 && mkdir -p /etc/bash_completion.d \
 # add npm completions
 && npm completion > /etc/bash_completion.d/npm \
 # add yarn completions
 && curl -fsSL --compressed \
    https://raw.githubusercontent.com/dsifford/yarn-completion/5bf2968493a7a76649606595cfca880a77e6ac0e/yarn-completion.bash \
  | tee /etc/bash_completion.d/yarn >/dev/null

# avoid "OSError: library nvvm not found" error
ENV CUDA_HOME="/usr/local/cuda"

SHELL ["/bin/bash", "-c"]

ENTRYPOINT ["docker-entrypoint.sh"]

USER node

CMD ["/bin/bash", "-l"]
