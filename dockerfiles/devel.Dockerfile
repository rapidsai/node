ARG BASE_IMAGE
ARG NODE_VERSION=15.14.0

FROM node:$NODE_VERSION-stretch-slim as node

FROM ${BASE_IMAGE}

ARG GCC_VERSION=9
ARG LLDB_VERSION=12
ARG CLANGD_VERSION=12
ARG CLANG_FORMAT_VERSION=12

ARG PARALLEL_LEVEL=4
ARG CMAKE_VERSION=3.20.2
ARG SCCACHE_VERSION=0.2.15

# Install dev dependencies and tools
RUN export DEBIAN_FRONTEND=noninteractive \
 && apt update -y \
 && apt install --no-install-recommends -y gpg wget software-properties-common \
 && add-apt-repository --no-update -y ppa:git-core/ppa \
 && add-apt-repository --no-update -y ppa:ubuntu-toolchain-r/test \
 # Install kitware CMake apt sources
 && wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null \
  | gpg --dearmor - \
  | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null \
 && bash -c 'echo -e "\
deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main\n\
" | tee /etc/apt/sources.list.d/kitware.list >/dev/null' \
 # Install LLVM apt sources
 && wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - \
 && bash -c 'echo -e "\
deb http://apt.llvm.org/$(lsb_release -cs)/ llvm-toolchain-$(lsb_release -cs) main\n\
deb-src  http://apt.llvm.org/$(lsb_release -cs)/ llvm-toolchain-$(lsb_release -cs) main\n\
" | tee /etc/apt/sources.list.d/llvm-dev.list >/dev/null' \
 && bash -c 'echo -e "\
deb http://apt.llvm.org/$(lsb_release -cs)/ llvm-toolchain-$(lsb_release -cs)-${LLDB_VERSION} main\n\
deb-src  http://apt.llvm.org/$(lsb_release -cs)/ llvm-toolchain-$(lsb_release -cs)-${LLDB_VERSION} main\n\
" | tee /etc/apt/sources.list.d/llvm-${LLDB_VERSION}.list >/dev/null' \
 && bash -c 'echo -e "\
deb http://apt.llvm.org/$(lsb_release -cs)/ llvm-toolchain-$(lsb_release -cs)-${CLANGD_VERSION} main\n\
deb-src  http://apt.llvm.org/$(lsb_release -cs)/ llvm-toolchain-$(lsb_release -cs)-${CLANGD_VERSION} main\n\
" | tee /etc/apt/sources.list.d/llvm-${CLANGD_VERSION}.list >/dev/null' \
 && bash -c 'echo -e "\
deb http://apt.llvm.org/$(lsb_release -cs)/ llvm-toolchain-$(lsb_release -cs)-${CLANG_FORMAT_VERSION} main\n\
deb-src  http://apt.llvm.org/$(lsb_release -cs)/ llvm-toolchain-$(lsb_release -cs)-${CLANG_FORMAT_VERSION} main\n\
" | tee /etc/apt/sources.list.d/llvm-${CLANG_FORMAT_VERSION}.list >/dev/null' \
 && apt update -y \
 && apt install --no-install-recommends -y \
    gcc-${GCC_VERSION} g++-${GCC_VERSION} \
    jq git entr nano sudo ninja-build bash-completion \
    # needed by cuda-gdb
    libtinfo5 libncursesw5 \
    # Install gdb, lldb (for llnode), and clangd for C++ intellisense and debugging in the container
    gdb lldb-${LLDB_VERSION} clangd-${CLANGD_VERSION} clang-format-${CLANG_FORMAT_VERSION} \
    # CMake
    curl libssl-dev libcurl4-openssl-dev zlib1g-dev \
    cmake=$(apt policy cmake 2>/dev/null | grep "$CMAKE_VERSION" | cut -d' ' -f6) \
    cmake-data=$(apt policy cmake 2>/dev/null | grep "$CMAKE_VERSION" | cut -d' ' -f6) \
    # X11 dependencies
    libxrandr-dev libxinerama-dev libxcursor-dev \
    # node-canvas dependencies
    libcairo2-dev libpango1.0-dev libjpeg-dev libgif-dev librsvg2-dev \
    # GLEW dependencies
    build-essential libxmu-dev libxi-dev libgl1-mesa-dev libegl1-mesa-dev libglu1-mesa-dev \
    # cuSpatial dependencies
    libgdal-dev \
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
 # Set alternatives for clangd
 && update-alternatives --remove-all clangd >/dev/null 2>&1 || true \
 && update-alternatives --install /usr/bin/clangd clangd /usr/bin/clangd-${CLANGD_VERSION} 100 \
 # Set clangd-${CLANGD_VERSION} as the default clangd
 && update-alternatives --set clangd /usr/bin/clangd-${CLANGD_VERSION} \
 # Set alternatives for clang-format
 && update-alternatives --remove-all clang-format >/dev/null 2>&1 || true \
 && update-alternatives --install /usr/bin/clang-format clang-format /usr/bin/clang-format-${CLANG_FORMAT_VERSION} 100 \
 # Set clang-format-${CLANG_FORMAT_VERSION} as the default clang-format
 && update-alternatives --set clang-format /usr/bin/clang-format-${CLANG_FORMAT_VERSION} \
 # Set alternatives for lldb and llvm-config so it's in the path for llnode
 && update-alternatives --remove-all lldb >/dev/null 2>&1 || true \
 && update-alternatives --remove-all llvm-config >/dev/null 2>&1 || true \
 && update-alternatives \
    --install /usr/bin/lldb lldb /usr/bin/lldb-${LLDB_VERSION} 100 \
    --slave /usr/bin/llvm-config llvm-config /usr/bin/llvm-config-${LLDB_VERSION} \
 # Set lldb-${LLDB_VERSION} as the default lldb, llvm-config-${LLDB_VERSION} as default llvm-config
 && update-alternatives --set lldb /usr/bin/lldb-${LLDB_VERSION} \
 # Install sccache
 && curl -o /tmp/sccache.tar.gz \
         -L "https://github.com/mozilla/sccache/releases/download/v$SCCACHE_VERSION/sccache-v$SCCACHE_VERSION-$(uname -m)-unknown-linux-musl.tar.gz" \
 && tar -C /tmp -xvf /tmp/sccache.tar.gz \
 && mv "/tmp/sccache-v$SCCACHE_VERSION-$(uname -m)-unknown-linux-musl/sccache" /bin/sccache \
 && chmod +x /bin/sccache \
 && cd / \
 # Clean up
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
 && echo node:node | chpasswd \
 && ln -s /usr/local/bin/node /usr/local/bin/nodejs \
 && ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
 && ln -s /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx \
 && npm install -g npm \
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
# Change the file location because certain bash sessions truncate .bash_history file upon close.\n\
# http://superuser.com/questions/575479/bash-history-truncated-to-500-lines-on-each-login\n\
export HISTFILE=/opt/node-rapids/modules/.cache/.eternal_bash_history;\n\
mkdir -p \$(dirname \$HISTFILE);\n\
touch \$HISTFILE;\n\
# flush commands to .bash_history immediately\n\
export PROMPT_COMMAND=\"history -a; \$PROMPT_COMMAND\";\n\
"' >> /home/node/.bashrc \
 && mkdir -p /etc/bash_completion.d \
 # add npm completions
 && npm completion > /etc/bash_completion.d/npm \
 # add yarn completions
 && curl -fsSL --compressed \
    https://raw.githubusercontent.com/dsifford/yarn-completion/5bf2968493a7a76649606595cfca880a77e6ac0e/yarn-completion.bash \
  | tee /etc/bash_completion.d/yarn >/dev/null \
 # globally install llnode
 && git clone --branch use-llvm-project-monorepo https://github.com/trxcllnt/llnode.git /usr/local/lib/llnode \
 && npm install --global --unsafe-perm --no-audit --no-fund /usr/local/lib/llnode \
 && which -a llnode

# avoid "OSError: library nvvm not found" error
ENV CUDA_HOME="/usr/local/cuda"

SHELL ["/bin/bash", "-c"]

ENTRYPOINT ["docker-entrypoint.sh"]

USER node

CMD ["/bin/bash", "-l"]
