ARG FROM_IMAGE
ARG FROM_IMAGE_DEFAULT
ARG DEVEL_IMAGE

FROM ${DEVEL_IMAGE} as devel

FROM ${FROM_IMAGE:-$FROM_IMAGE_DEFAULT}

SHELL ["/bin/bash", "-c"]

ENV NVIDIA_DRIVER_CAPABILITIES all

ENV CUDA_HOME="/usr/local/cuda"
ENV LD_LIBRARY_PATH="\
/usr/lib/aarch64-linux-gnu:\
/usr/lib/x86_64-linux-gnu:\
/usr/lib/i386-linux-gnu:\
${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}\
${CUDA_HOME}/lib:\
${CUDA_HOME}/lib64:\
/usr/local/lib:\
/usr/lib"

ADD --chown=root:root https://gitlab.com/nvidia/container-images/opengl/-/raw/5191cf205d3e4bb1150091f9464499b076104354/glvnd/runtime/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# Install gcc-9 toolchain
RUN export DEBIAN_FRONTEND=noninteractive \
 && apt update \
 && apt install --no-install-recommends -y \
    software-properties-common \
 && add-apt-repository --no-update -y ppa:ubuntu-toolchain-r/test \
 && apt update \
 && apt install --no-install-recommends -y \
    libstdc++6 \
    # From opengl/glvnd:runtime
    libxau6 libxdmcp6 libxcb1 libxext6 libx11-6 \
    libglvnd0 libopengl0 libgl1 libglx0 libegl1 libgles2 \
 \
 && chmod 0644 /usr/share/glvnd/egl_vendor.d/10_nvidia.json \
 && echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf \
 && echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf \
 # Clean up
 && add-apt-repository --remove -y ppa:ubuntu-toolchain-r/test \
 && apt remove -y software-properties-common \
 && apt autoremove -y && apt clean \
 && rm -rf \
    /tmp/* \
    /var/tmp/* \
    /var/lib/apt/lists/* \
    /var/cache/apt/archives/*

# Install node
COPY --from=devel /usr/local/bin/node                 /usr/local/bin/node
COPY --from=devel /usr/local/include/node             /usr/local/include/node
COPY --from=devel /usr/local/lib/node_modules         /usr/local/lib/node_modules
# Install yarn
COPY --from=devel /usr/local/bin/yarn                 /usr/local/bin/yarn
COPY --from=devel /usr/local/bin/yarn.js              /usr/local/bin/yarn.js
COPY --from=devel /usr/local/bin/yarn.cmd             /usr/local/bin/yarn.cmd
COPY --from=devel /usr/local/bin/yarnpkg              /usr/local/bin/yarnpkg
COPY --from=devel /usr/local/bin/yarnpkg.cmd          /usr/local/bin/yarnpkg.cmd
COPY --from=devel /usr/local/lib/cli.js               /usr/local/lib/cli.js
COPY --from=devel /usr/local/lib/v8-compile-cache.js  /usr/local/lib/v8-compile-cache.js
# Copy entrypoint
COPY --from=devel /usr/local/bin/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

ARG UID=1000
ARG ADDITIONAL_GROUPS=

RUN useradd --uid $UID --user-group ${ADDITIONAL_GROUPS} --shell /bin/bash --create-home node \
 && ln -s /usr/local/bin/node /usr/local/bin/nodejs \
 && ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
 && ln -s /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx \
 # smoke tests
 && node --version && npm --version && yarn --version

ENV NODE_OPTIONS="--experimental-vm-modules --trace-uncaught"

WORKDIR /home/node

ENTRYPOINT ["docker-entrypoint.sh"]

SHELL ["/bin/bash", "-l"]

CMD ["node"]
