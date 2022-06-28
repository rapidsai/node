# syntax=docker/dockerfile:1.3

ARG AMD64_BASE
ARG ARM64_BASE
ARG NODE_VERSION=16.15.1

FROM node:$NODE_VERSION-bullseye-slim as node

FROM ${AMD64_BASE} as base-amd64

FROM ${ARM64_BASE} as base-arm64

ONBUILD RUN \
    if [[ -d /usr/local/cuda/lib64 ] && [ ! -f /usr/local/cuda/lib64/libcudart.so ]]; then \
        minor="$(nvcc --version | head -n4 | tail -n1 | cut -d' ' -f5 | cut -d',' -f1)"; \
        major="$(nvcc --version | head -n4 | tail -n1 | cut -d' ' -f5 | cut -d',' -f1 | cut -d'.' -f1)"; \
        ln -s /usr/local/cuda/lib64/libcudart.so.$minor /usr/local/cuda/lib64/libcudart.so.$major; \
        ln -s /usr/local/cuda/lib64/libcudart.so.$major /usr/local/cuda/lib64/libcudart.so; \
        rm /etc/ld.so.cache && ldconfig; \
    fi

FROM base-${TARGETARCH} as compilers

SHELL ["/bin/bash", "-c"]

ENV CUDA_HOME="/usr/local/cuda"
ENV PATH="$PATH:\
${CUDA_HOME}/bin:\
${CUDA_HOME}/nvvm/bin"
ENV LD_LIBRARY_PATH="\
/usr/lib/aarch64-linux-gnu:\
/usr/lib/x86_64-linux-gnu:\
/usr/lib/i386-linux-gnu:\
${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}\
${CUDA_HOME}/lib64:\
${CUDA_HOME}/nvvm/lib64:\
${CUDA_HOME}/lib64/stubs"

ARG GCC_VERSION=9
ARG SCCACHE_VERSION=0.2.15
ARG LINUX_VERSION=ubuntu20.04

ARG NODE_VERSION=16.15.1
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

ADD --chown=root:root https://gitlab.com/nvidia/container-images/opengl/-/raw/5191cf205d3e4bb1150091f9464499b076104354/glvnd/runtime/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# Install compilers
RUN export DEBIAN_FRONTEND=noninteractive \
 && apt update \
 && apt install --no-install-recommends -y \
    gpg wget software-properties-common lsb-release \
 && add-apt-repository --no-update -y ppa:git-core/ppa \
 && add-apt-repository --no-update -y ppa:ubuntu-toolchain-r/test \
 # Install kitware cmake apt repository
 && wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null \
  | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null \
 && bash -c 'echo -e "\
deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main\n\
" | tee /etc/apt/sources.list.d/kitware.list >/dev/null' \
 \
 && apt update \
 && apt install --no-install-recommends -y \
    git cmake ninja-build \
    gcc-${GCC_VERSION} g++-${GCC_VERSION} gdb \
    curl libssl-dev libcurl4-openssl-dev xz-utils zlib1g-dev liblz4-dev \
    # From opengl/glvnd:devel
    pkg-config \
    libxau6 libxdmcp6 libxcb1 libxext6 libx11-6 \
    libglvnd-dev libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev \
 \
 && chmod 0644 /usr/share/glvnd/egl_vendor.d/10_nvidia.json \
 && echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf \
 && echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf \
 \
 # Install sccache
 && curl -SsL "https://github.com/mozilla/sccache/releases/download/v$SCCACHE_VERSION/sccache-v$SCCACHE_VERSION-$(uname -m)-unknown-linux-musl.tar.gz" \
    | tar -C /usr/bin -zf - --wildcards --strip-components=1 -x */sccache \
 && chmod +x /usr/bin/sccache \
 \
 # Install npm
 && bash -c 'echo -e "\
fund=false\n\
audit=false\n\
save-prefix=\n\
--omit=optional\n\
save-exact=true\n\
package-lock=false\n\
update-notifier=false\n\
scripts-prepend-node-path=true\n\
registry=https://registry.npmjs.org/\n\
" | tee /root/.npmrc >/dev/null' \
 && ln -s /usr/local/bin/node /usr/local/bin/nodejs \
 && ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
 && ln -s /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx \
 # Smoke tests
 && echo "node version: $(node --version)" \
 && echo " npm version: $(npm --version)" \
 && echo "yarn version: $(yarn --version)" \
 \
 # Clean up
 && add-apt-repository --remove -y ppa:git-core/ppa \
 && add-apt-repository --remove -y ppa:ubuntu-toolchain-r/test \
 && apt autoremove -y && apt clean \
 && rm -rf \
    /tmp/* \
    /var/tmp/* \
    /var/cache/apt/* \
    /var/lib/apt/lists/*

ENTRYPOINT ["docker-entrypoint.sh"]

WORKDIR /

FROM compilers as main-arm64

ONBUILD ARG ADDITIONAL_GROUPS="--groups sudo,video"

FROM compilers as main-amd64

ONBUILD ARG LLDB_VERSION=12
ONBUILD ARG CLANGD_VERSION=12
ONBUILD ARG CLANG_FORMAT_VERSION=12

# Install dependencies and dev tools (llnode etc.)
ONBUILD RUN export DEBIAN_FRONTEND=noninteractive \
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
    # lldb (for llnode)
    lldb-${LLDB_VERSION} libllvm${LLDB_VERSION} \
    # clangd for C++ intellisense and debugging
    clangd-${CLANGD_VERSION} \
    # clang-format for automatically formatting C++ and TS/JS
    clang-format-${CLANG_FORMAT_VERSION} \
 \
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
 # Globally install llnode
 && mkdir -p /usr/local/lib/llnode \
 && wget -O - https://github.com/trxcllnt/llnode/archive/refs/heads/use-llvm-project-monorepo.tar.gz \
  | tar -C /usr/local/lib/llnode -xzf - --strip-components=1 \
 && npm pack --pack-destination /usr/local/lib/llnode /usr/local/lib/llnode \
 && npm install --location=global --unsafe-perm --no-audit --no-fund --no-update-notifier /usr/local/lib/llnode/llnode-*.tgz \
 && echo "llnode: $(which -a llnode)" \
 && echo "llnode version: $(llnode --version)" \
 \
 # Clean up
 && apt autoremove -y && apt clean \
 && rm -rf \
    /tmp/* \
    /var/tmp/* \
    /var/cache/apt/* \
    /var/lib/apt/lists/* \
    /usr/local/lib/llnode \
    /etc/apt/sources.list.d/llvm-${LLDB_VERSION}.list \
    /etc/apt/sources.list.d/llvm-${CLANGD_VERSION}.list \
    /etc/apt/sources.list.d/llvm-${CLANG_FORMAT_VERSION}.list

FROM main-${TARGETARCH}

ENV NVIDIA_DRIVER_CAPABILITIES all

ARG TARGETARCH

ARG ADDITIONAL_GROUPS
ARG UCX_VERSION=1.12.1
ARG FIXUID_VERSION=0.5.1
ARG NODE_WEBRTC_VERSION=0.4.7

# Install dependencies (llnode etc.)
RUN export DEBIAN_FRONTEND=noninteractive \
 && apt update \
 && apt install --no-install-recommends -y \
    jq entr ssh vim nano sudo less bash-completion \
    # X11 dependencies
    libxi-dev libxrandr-dev libxinerama-dev libxcursor-dev \
    # node-canvas dependencies
    libcairo2-dev libpango1.0-dev libjpeg-dev libgif-dev librsvg2-dev \
    # GLFW Wayland dependencies
    extra-cmake-modules libwayland-dev wayland-protocols libxkbcommon-dev \
    # GLEW dependencies
    build-essential libxmu-dev libgl1-mesa-dev libegl1-mesa-dev libglu1-mesa-dev \
    # cuSpatial dependencies
    libgdal-dev \
    # SQL dependencies
    maven openjdk-8-jdk-headless openjdk-8-jre-headless libboost-regex-dev libboost-system-dev libboost-filesystem-dev \
    # UCX build dependencies
    # automake autoconf libtool \
    # UCX runtime dependencies
    libibverbs-dev librdmacm-dev libnuma-dev \
 \
 # Install UCX
 && wget -O /var/cache/apt/archives/ucx-v${UCX_VERSION}-${LINUX_VERSION}-mofed5-cuda11.deb \
    https://github.com/openucx/ucx/releases/download/v${UCX_VERSION}/ucx-v${UCX_VERSION}-${LINUX_VERSION}-mofed5-cuda11.deb \
 && dpkg -i /var/cache/apt/archives/ucx-v${UCX_VERSION}-${LINUX_VERSION}-mofed5-cuda11.deb || true && apt --fix-broken install -y \
 \
 # Install fixuid
 && curl -SsL "https://github.com/boxboat/fixuid/releases/download/v$FIXUID_VERSION/fixuid-$FIXUID_VERSION-linux-${TARGETARCH}.tar.gz" \
  | tar -C /usr/bin -xzf - \
 && chown root:root /usr/bin/fixuid && chmod 4755 /usr/bin/fixuid && mkdir -p /etc/fixuid \
 && bash -c 'echo -e "\
user: rapids\n\
group: rapids\n\
paths:\n\
  - /opt/rapids\n\
  - /opt/rapids/node\n\
" | tee /etc/fixuid/config.yml >/dev/null' \
 \
 # Add a non-root user
 && useradd \
    --uid 1000 --shell /bin/bash \
    --user-group ${ADDITIONAL_GROUPS} \
    --create-home --home-dir /opt/rapids \
    rapids \
 && mkdir -p /opt/rapids/node/.cache \
 && cp /root/.npmrc /opt/rapids/.npmrc \
 && ln -s /opt/rapids/node/.vscode/server /opt/rapids/.vscode-server \
 && ln -s /opt/rapids/node/.vscode/server-insiders /opt/rapids/.vscode-server-insiders \
 && chown -R rapids:rapids /opt/rapids \
 && bash -c 'echo "rapids ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/nopasswd' \
 \
 # yellow + blue terminal prompt
 && sed -ri "s/32m/33m/g" /opt/rapids/.bashrc \
 && sed -ri "s/34m/36m/g" /opt/rapids/.bashrc \
 # Persist infinite bash history on the host
 && bash -c 'echo -e "\
\n\
# Infinite bash history\n\
export HISTSIZE=-1;\n\
export HISTFILESIZE=-1;\n\
export HISTCONTROL=ignoreboth;\n\
\n\
# Change the file location because certain bash sessions truncate .bash_history file upon close.\n\
# http://superuser.com/questions/575479/bash-history-truncated-to-500-lines-on-each-login\n\
export HISTFILE=/opt/rapids/node/.cache/.eternal_bash_history;\n\
\n\
mkdir -p \$(dirname \$HISTFILE) && touch \$HISTFILE;\n\
mkdir -p /opt/rapids/node/.vscode/server{,-insiders}\n\
\n\
# flush commands to .bash_history immediately\n\
export PROMPT_COMMAND=\"history -a; \$PROMPT_COMMAND\";\n\
"' >> /opt/rapids/.bashrc \
 \
 # Add npm and yarn completions
 && mkdir -p /etc/bash_completion.d \
 && npm completion > /etc/bash_completion.d/npm \
 && curl -fsSL --compressed \
    https://raw.githubusercontent.com/dsifford/yarn-completion/5bf2968493a7a76649606595cfca880a77e6ac0e/yarn-completion.bash \
  | tee /etc/bash_completion.d/yarn >/dev/null \
 \
 # Install NVENC-enabled wrtc
 && wget -O /opt/rapids/wrtc-dev.tgz \
    https://github.com/trxcllnt/node-webrtc-builds/releases/download/v${NODE_WEBRTC_VERSION}/wrtc-${NODE_WEBRTC_VERSION}-linux-${TARGETARCH}.tgz \
 && npm install --location=global --unsafe-perm --no-audit --no-fund --no-update-notifier /opt/rapids/wrtc-dev.tgz \
 # Clean up
 && apt autoremove -y && apt clean \
 && rm -rf \
    /tmp/* \
    /var/tmp/* \
    /var/cache/apt/* \
    /var/lib/apt/lists/*

ENV NO_UPDATE_NOTIFIER=1
ENV NODE_PATH=/usr/local/lib/node_modules
ENV NODE_OPTIONS="--experimental-vm-modules --trace-uncaught"

USER rapids

WORKDIR /opt/rapids/node

ENTRYPOINT ["fixuid", "-q", "docker-entrypoint.sh"]

CMD ["/bin/bash", "-l"]
