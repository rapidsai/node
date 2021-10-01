ARG FROM_IMAGE
ARG FROM_IMAGE_DEFAULT
ARG NODE_VERSION=16.9.1

FROM node:$NODE_VERSION-stretch-slim as node

FROM ${FROM_IMAGE:-$FROM_IMAGE_DEFAULT}

SHELL ["/bin/bash", "-c"]

ENV CUDA_HOME="/usr/local/cuda"
ENV LD_LIBRARY_PATH="\
${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}\
${CUDA_HOME}/lib:\
${CUDA_HOME}/lib64:\
/usr/local/lib:\
/usr/lib"

ARG GCC_VERSION=9
ARG LLDB_VERSION=12
ARG CLANGD_VERSION=12
ARG CMAKE_VERSION=3.21.3
ARG SCCACHE_VERSION=0.2.15
ARG CLANG_FORMAT_VERSION=12

# Install dev dependencies and tools
RUN export DEBIAN_FRONTEND=noninteractive \
 && apt update \
 && apt install --no-install-recommends -y \
    gpg wget software-properties-common \
 && add-apt-repository --no-update -y ppa:git-core/ppa \
 && add-apt-repository --no-update -y ppa:ubuntu-toolchain-r/test \
 # Install LLVM apt sources
 && wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - \
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
 \
 && apt update \
 && apt install --no-install-recommends -y \
    gcc-${GCC_VERSION} g++-${GCC_VERSION} \
    jq git entr nano sudo ninja-build bash-completion \
    # Install gdb
    gdb \
    # lldb (for llnode)
    lldb-${LLDB_VERSION} libllvm${LLDB_VERSION} \
    # clangd for C++ intellisense and debugging
    clangd-${CLANGD_VERSION} \
    # clang-format for automatically formatting C++ and TS/JS
    clang-format-${CLANG_FORMAT_VERSION} \
    # CMake
    curl libssl-dev libcurl4-openssl-dev zlib1g-dev liblz4-dev \
 \
 # Remove any existing gcc and g++ alternatives
 && (update-alternatives --remove-all cc  >/dev/null 2>&1 || true) \
 && (update-alternatives --remove-all c++ >/dev/null 2>&1 || true) \
 && (update-alternatives --remove-all gcc >/dev/null 2>&1 || true) \
 && (update-alternatives --remove-all g++ >/dev/null 2>&1 || true) \
 && (update-alternatives --remove-all gcov >/dev/null 2>&1 || true) \
 && update-alternatives \
    --install /usr/bin/gcc gcc /usr/bin/gcc-${GCC_VERSION} 100 \
    --slave /usr/bin/cc cc /usr/bin/gcc-${GCC_VERSION} \
    --slave /usr/bin/g++ g++ /usr/bin/g++-${GCC_VERSION} \
    --slave /usr/bin/c++ c++ /usr/bin/g++-${GCC_VERSION} \
    --slave /usr/bin/gcov gcov /usr/bin/gcov-${GCC_VERSION} \
 # Set gcc-${GCC_VERSION} as the default gcc
 && update-alternatives --set gcc /usr/bin/gcc-${GCC_VERSION} \
 # Set alternatives for clangd
 && (update-alternatives --remove-all clangd >/dev/null 2>&1 || true) \
 && update-alternatives --install /usr/bin/clangd clangd /usr/bin/clangd-${CLANGD_VERSION} 100 \
 # Set clangd-${CLANGD_VERSION} as the default clangd
 && update-alternatives --set clangd /usr/bin/clangd-${CLANGD_VERSION} \
 # Set alternatives for clang-format
 && (update-alternatives --remove-all clang-format >/dev/null 2>&1 || true) \
 && update-alternatives --install /usr/bin/clang-format clang-format /usr/bin/clang-format-${CLANG_FORMAT_VERSION} 100 \
 # Set clang-format-${CLANG_FORMAT_VERSION} as the default clang-format
 && update-alternatives --set clang-format /usr/bin/clang-format-${CLANG_FORMAT_VERSION} \
 # Set alternatives for lldb and llvm-config so it's in the path for llnode
 && (update-alternatives --remove-all lldb >/dev/null 2>&1 || true) \
 && (update-alternatives --remove-all llvm-config >/dev/null 2>&1 || true) \
 && update-alternatives \
    --install /usr/bin/lldb lldb /usr/bin/lldb-${LLDB_VERSION} 100 \
    --slave /usr/bin/llvm-config llvm-config /usr/bin/llvm-config-${LLDB_VERSION} \
 # Set lldb-${LLDB_VERSION} as the default lldb, llvm-config-${LLDB_VERSION} as default llvm-config
 && update-alternatives --set lldb /usr/bin/lldb-${LLDB_VERSION} \
 \
 # Install CMake
 && curl -fsSL --compressed -o "/tmp/cmake-$CMAKE_VERSION-linux-$(uname -m).sh" \
    "https://github.com/Kitware/CMake/releases/download/v$CMAKE_VERSION/cmake-$CMAKE_VERSION-linux-$(uname -m).sh" \
 && sh "/tmp/cmake-$CMAKE_VERSION-linux-$(uname -m).sh" --skip-license --exclude-subdir --prefix=/usr/local \
 \
 # Install sccache
 && curl -o /tmp/sccache.tar.gz \
         -L "https://github.com/mozilla/sccache/releases/download/v$SCCACHE_VERSION/sccache-v$SCCACHE_VERSION-$(uname -m)-unknown-linux-musl.tar.gz" \
 && tar -C /tmp -xvf /tmp/sccache.tar.gz \
 && mv "/tmp/sccache-v$SCCACHE_VERSION-$(uname -m)-unknown-linux-musl/sccache" /usr/bin/sccache \
 && chmod +x /usr/bin/sccache \
 && cd / \
 \
 # Clean up
 && apt autoremove -y && apt clean \
 && rm -rf \
    /tmp/* \
    /var/tmp/* \
    /var/lib/apt/lists/* \
    /etc/apt/sources.list.d/llvm-${LLDB_VERSION}.list \
    /etc/apt/sources.list.d/llvm-${CLANGD_VERSION}.list \
    /etc/apt/sources.list.d/llvm-${CLANG_FORMAT_VERSION}.list

ARG NODE_VERSION
ENV NODE_VERSION=$NODE_VERSION

# Install node
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/include/node /usr/local/include/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
# Install yarn
COPY --from=node /opt/yarn-v*/bin/* /usr/local/bin/
COPY --from=node /opt/yarn-v*/lib/* /usr/local/lib/
# Copy entrypoint
COPY --from=node /usr/local/bin/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

ARG UID=1000
ARG ADDITIONAL_GROUPS=

RUN useradd \
    --uid $UID \
    --user-group ${ADDITIONAL_GROUPS} \
    --shell /bin/bash --create-home node \
 && mkdir -p /opt/rapids/node/modules/.cache \
 && chown -R node:node /opt/rapids/node \
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
export HISTFILE=/opt/rapids/node/modules/.cache/.eternal_bash_history;\n\
mkdir -p \$(dirname \$HISTFILE) && touch \$HISTFILE;\n\
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

ENTRYPOINT ["docker-entrypoint.sh"]

WORKDIR /opt/rapids/node

CMD ["/bin/bash", "-l"]
