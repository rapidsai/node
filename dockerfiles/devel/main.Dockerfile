# syntax=docker/dockerfile:1.3

ARG FROM_IMAGE
ARG FROM_IMAGE_DEFAULT
ARG NODE_VERSION=16.10.0

FROM node:$NODE_VERSION-stretch-slim as node

FROM ${FROM_IMAGE:-$FROM_IMAGE_DEFAULT} as compilers

SHELL ["/bin/bash", "-c"]

ENV CUDA_HOME="/usr/local/cuda"
ENV LD_LIBRARY_PATH="\
${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}\
${CUDA_HOME}/lib:\
${CUDA_HOME}/lib64:\
/usr/local/lib:\
/usr/lib"

ARG GCC_VERSION=9
ARG CMAKE_VERSION=3.21.3
ARG SCCACHE_VERSION=0.2.15

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

# https://github.com/moby/buildkit/blob/b8462c3b7c15b14a8c30a79fad298a1de4ca9f74/frontend/dockerfile/docs/syntax.md#example-cache-apt-packages
RUN --mount=type=cache,target=/var/lib/apt \
    --mount=type=cache,target=/var/cache/apt \
    rm -f /etc/apt/apt.conf.d/docker-clean; \
    echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache; \
 \
 # Install compilers
    export DEBIAN_FRONTEND=noninteractive \
 && apt update \
 && apt install --no-install-recommends -y \
    gpg wget software-properties-common \
 && add-apt-repository --no-update -y ppa:git-core/ppa \
 && add-apt-repository --no-update -y ppa:ubuntu-toolchain-r/test \
 \
 && apt update \
 && apt install --no-install-recommends -y \
    git ninja-build \
    gcc-${GCC_VERSION} g++-${GCC_VERSION} gdb \
    # CMake dependencies
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
 \
 # Install CMake
 && curl -fsSL --compressed -o "/tmp/cmake-$CMAKE_VERSION-linux-$(uname -m).sh" \
    "https://github.com/Kitware/CMake/releases/download/v$CMAKE_VERSION/cmake-$CMAKE_VERSION-linux-$(uname -m).sh" \
 && sh "/tmp/cmake-$CMAKE_VERSION-linux-$(uname -m).sh" --skip-license --exclude-subdir --prefix=/usr/local \
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
optional=false\n\
save-exact=true\n\
package-lock=false\n\
update-notifier=false\n\
scripts-prepend-node-path=true\n\
registry=https://registry.npmjs.org/\n\
" | tee /root/.npmrc >/dev/null' \
 && ln -s /usr/local/bin/node /usr/local/bin/nodejs \
 && ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
 && ln -s /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx \
 && npm install --global --unsafe-perm --no-audit --no-fund npm \
 # Smoke tests
 && node --version && npm --version && yarn --version \
 \
 # Clean up
 && add-apt-repository --remove -y ppa:git-core/ppa \
 && add-apt-repository --remove -y ppa:ubuntu-toolchain-r/test \
 && apt autoremove -y && apt clean \
 && rm -rf /tmp/* /var/tmp/*

ENTRYPOINT ["docker-entrypoint.sh"]

WORKDIR /

FROM ${FROM_IMAGE:-$FROM_IMAGE_DEFAULT} as nvenc

RUN apt update \
 && NVENC_VERSION=$(apt policy libnvidia-encode-* 2>/dev/null | grep libnvidia-encode- | cut -d'-' -f3 | cut -d':' -f1 | sort -rh | head -n1) \
 && DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends libnvidia-encode-$NVENC_VERSION \
 && apt autoremove -y && apt clean && rm -rf /tmp/* /var/tmp/* /var/lib/apt/lists/* /var/cache/apt/archives/* \
 && ln -s /usr/lib/$(uname -m)-linux-gnu/libcuda.so /usr/local/lib/libcuda.so \
 && ln -s /usr/lib/$(uname -m)-linux-gnu/libnvcuvid.so /usr/local/lib/libnvcuvid.so \
 && ln -s /usr/lib/$(uname -m)-linux-gnu/libnvidia-encode.so /usr/local/lib/libnvidia-encode.so

FROM compilers as wrtc

ARG SCCACHE_REGION
ARG SCCACHE_BUCKET
ARG SCCACHE_IDLE_TIMEOUT
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY

RUN --mount=type=secret,id=AWS_ACCESS_KEY_ID \
    --mount=type=secret,id=AWS_SECRET_ACCESS_KEY \
    --mount=type=cache,target=/opt/node-webrtc \
    --mount=type=bind,from=nvenc,source=/usr/local/lib/libcuda.so,target=/usr/local/lib/libcuda.so \
    --mount=type=bind,from=nvenc,source=/usr/local/lib/libnvcuvid.so,target=/usr/local/lib/libnvcuvid.so \
    --mount=type=bind,from=nvenc,source=/usr/local/lib/libnvidia-encode.so,target=/usr/local/lib/libnvidia-encode.so \
    \
    if [ ! -f /opt/node-webrtc/build/Release/wrtc.node ]; then \
        export AWS_ACCESS_KEY_ID="$(cat /run/secrets/AWS_ACCESS_KEY_ID 2>/dev/null || echo $AWS_ACCESS_KEY_ID)"; \
        export AWS_SECRET_ACCESS_KEY="$(cat /run/secrets/AWS_SECRET_ACCESS_KEY 2>/dev/null || echo $AWS_SECRET_ACCESS_KEY)"; \
        apt update \
         && DEBIAN_FRONTEND=noninteractive \
            apt install -y --no-install-recommends python \
         && apt autoremove -y && apt clean \
         && rm -rf \
            /tmp/* \
            /var/tmp/* \
            /var/lib/apt/lists/* \
            /var/cache/apt/archives/* \
         && git clone --depth 1 --branch node-webrtc-nvenc \
            https://github.com/trxcllnt/node-webrtc.git \
            /opt/node-webrtc-nvenc \
         && cd /opt/node-webrtc-nvenc \
         && env SKIP_DOWNLOAD=1 \
            CMAKE_MESSAGE_LOG_LEVEL=VERBOSE \
            CMAKE_C_COMPILER_LAUNCHER=/usr/bin/sccache \
            CMAKE_CXX_COMPILER_LAUNCHER=/usr/bin/sccache \
            CMAKE_CUDA_COMPILER_LAUNCHER=/usr/bin/sccache \
            npm install --no-audit --no-fund \
         && cd / \
         \
         && mkdir -p /opt/node-webrtc/build \
         && cp -R /opt/node-webrtc-nvenc/lib /opt/node-webrtc/ \
         && cp -R /opt/node-webrtc-nvenc/build/Release /opt/node-webrtc/build/ \
         && cp -R /opt/node-webrtc-nvenc/{README,LICENSE,THIRD_PARTY_LICENSES}.md /opt/node-webrtc/ \
         && bash -c 'echo -e "{\n\
         \"name\": \"wrtc\",\n\
         \"version\": \"0.4.7-dev\",\n\
         \"author\": \"Alan K <ack@modeswitch.org> (http://blog.modeswitch.org)\",\n\
         \"homepage\": \"https://github.com/node-webrtc/node-webrtc\",\n\
         \"bugs\": \"https://github.com/node-webrtc/node-webrtc/issues\",\n\
         \"license\": \"BSD-2-Clause\",\n\
         \"main\": \"lib/index.js\",\n\
         \"browser\": \"lib/browser.js\",\n\
         \"repository\": {\n\
             \"type\": \"git\",\n\
             \"url\": \"http://github.com/node-webrtc/node-webrtc.git\"\n\
         },\n\
         \"files\": [\n\
             \"lib\",\n\
             \"build\",\n\
             \"README.md\",\n\
             \"LICENSE.md\",\n\
             \"THIRD_PARTY_LICENSES.md\"\n\
         ]\n\
         }\n\
         " | tee /opt/node-webrtc/package.json >/dev/null'; \
    fi; \
    mkdir -p /opt/rapids; \
    npm pack --pack-destination /opt/rapids /opt/node-webrtc; \
 \
 # Install NVENC-enabled wrtc
 npm install --global --unsafe-perm --no-audit --no-fund /opt/rapids/wrtc-0.4.7-dev.tgz

FROM compilers

ENV NVIDIA_DRIVER_CAPABILITIES all

ARG LLDB_VERSION=12
ARG CLANGD_VERSION=12
ARG FIXUID_VERSION=0.5.1
ARG CLANG_FORMAT_VERSION=12

ARG ADDITIONAL_GROUPS=

# https://github.com/moby/buildkit/blob/b8462c3b7c15b14a8c30a79fad298a1de4ca9f74/frontend/dockerfile/docs/syntax.md#example-cache-apt-packages
RUN --mount=type=cache,target=/var/lib/apt \
    --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/usr/local/lib/llnode \
    rm -f /etc/apt/apt.conf.d/docker-clean; \
    echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache; \
 \
 # Install dependencies and dev tools (llnode etc.)
    export DEBIAN_FRONTEND=noninteractive \
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
    jq entr nano sudo bash-completion \
    # lldb (for llnode)
    lldb-${LLDB_VERSION} libllvm${LLDB_VERSION} \
    # clangd for C++ intellisense and debugging
    clangd-${CLANGD_VERSION} \
    # clang-format for automatically formatting C++ and TS/JS
    clang-format-${CLANG_FORMAT_VERSION} \
    # X11 dependencies
    libxi-dev libxrandr-dev libxinerama-dev libxcursor-dev \
    # node-canvas dependencies
    libcairo2-dev libpango1.0-dev libjpeg-dev libgif-dev librsvg2-dev \
    # GLFW Wayland dependencies
    extra-cmake-modules libwayland-dev wayland-protocols libxkbcommon-dev \
    # GLEW dependencies
    build-essential libxmu-dev libxi-dev libgl1-mesa-dev libegl1-mesa-dev libglu1-mesa-dev \
    # cuSpatial dependencies
    libgdal-dev \
    # SQL dependencies
    maven openjdk-8-jdk libboost-regex-dev libboost-system-dev libboost-filesystem-dev \
    # UCX build dependencies
    automake autoconf libtool \
    # UCX runtime dependencies
    libibverbs-dev librdmacm-dev libnuma-dev libhwloc-dev \
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
 # Install UCX
 && git clone --depth 1 --branch v1.11.x https://github.com/openucx/ucx.git /tmp/ucx \
 && cd /tmp/ucx \
 && sed -i 's/io_demo_LDADD =/io_demo_LDADD = $(CUDA_LDFLAGS)/' /tmp/ucx/test/apps/iodemo/Makefile.am \
 && /tmp/ucx/autogen.sh && mkdir /tmp/ucx/build && cd /tmp/ucx/build \
 && ../contrib/configure-release \
    --prefix=/usr/local \
    --without-java --with-cuda=/usr/local/cuda \
    --enable-mt CPPFLAGS=-I/usr/local/cuda/include \
 && make -C /tmp/ucx/build -j install \
 \
 # Install fixuid
 && curl -SsL "https://github.com/boxboat/fixuid/releases/download/v$FIXUID_VERSION/fixuid-$FIXUID_VERSION-linux-$(dpkg-architecture -q DEB_BUILD_ARCH).tar.gz" \
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
 # Globally install llnode
 && mkdir -p /usr/local/lib/llnode && cd /usr/local/lib/llnode \
 && (git init 2>/dev/null || true) \
 && (git remote add origin https://github.com/trxcllnt/llnode.git 2>/dev/null || true) \
 && git fetch origin use-llvm-project-monorepo && git checkout use-llvm-project-monorepo \
 && cd / \
 && npm install --global --unsafe-perm --no-audit --no-fund /usr/local/lib/llnode \
 && which -a llnode \
 \
 # Clean up
 && apt autoremove -y && apt clean \
 && rm -rf /tmp/* /var/tmp/* \
    /etc/apt/sources.list.d/llvm-${LLDB_VERSION}.list \
    /etc/apt/sources.list.d/llvm-${CLANGD_VERSION}.list \
    /etc/apt/sources.list.d/llvm-${CLANG_FORMAT_VERSION}.list

ENV NODE_PATH=/usr/local/lib/node_modules

COPY --from=wrtc --chown=root:root /usr/local/lib/node_modules /usr/local/lib/node_modules
COPY --from=wrtc --chown=rapids:rapids /opt/rapids/wrtc-0.4.7-dev.tgz /opt/rapids/wrtc-0.4.7-dev.tgz

USER rapids

WORKDIR /opt/rapids/node

ENTRYPOINT ["fixuid", "-q", "docker-entrypoint.sh"]

CMD ["/bin/bash", "-l"]
