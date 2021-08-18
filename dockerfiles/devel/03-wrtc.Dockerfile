#syntax=docker/dockerfile:1.2

ARG FROM_IMAGE

FROM ${FROM_IMAGE}

USER root

RUN export DEBIAN_FRONTEND=noninteractive \
 && apt update --fix-missing \
 && apt install -y --no-install-recommends \
    libxi-dev libxrandr-dev python3 libnvidia-encode-470 \
 && update-alternatives --install /usr/bin/python python $(realpath $(which python3)) 1 \
 && update-alternatives --set python $(realpath $(which python3)) \
 # Clean up
 && apt autoremove -y && apt clean \
 && rm -rf \
    /tmp/* \
    /var/tmp/* \
    /var/lib/apt/lists/*

COPY --chown=root:root .npmrc /root/.npmrc

ARG SCCACHE_REGION
ARG SCCACHE_BUCKET
ARG SCCACHE_CACHE_SIZE
ARG SCCACHE_IDLE_TIMEOUT

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY

RUN --mount=type=secret,id=aws_creds \
 [[ -f /run/secrets/aws_creds ]] && \
    set -a && . /run/secrets/aws_creds && set +a \
    || true \
 && mkdir -p /opt/node-webrtc/build \
 && git clone --branch node-webrtc-nvenc \
    https://github.com/trxcllnt/node-webrtc.git \
    /opt/node-webrtc-nvenc \
 && cd /opt/node-webrtc-nvenc \
 && env \
    SKIP_DOWNLOAD=1 \
    CMAKE_C_COMPILER_LAUNCHER=/usr/bin/sccache \
    CMAKE_CXX_COMPILER_LAUNCHER=/usr/bin/sccache \
    CMAKE_CUDA_COMPILER_LAUNCHER=/usr/bin/sccache \
    npm install --no-audit --no-fund \
 && mv /opt/node-webrtc-nvenc/lib /opt/node-webrtc/ \
 && mv /opt/node-webrtc-nvenc/build/Release /opt/node-webrtc/build/ \
 && mv /opt/node-webrtc-nvenc/{README,LICENSE,THIRD_PARTY_LICENSES}.md /opt/node-webrtc/ \
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
" | tee /opt/node-webrtc/package.json >/dev/null' \
 && cd / \
 && npm pack --pack-destination /home/node /opt/node-webrtc \
 && rm -rf /opt/node-webrtc-nvenc /opt/node-webrtc \
 && chown -R node:node /home/node \
 && npm install --global --unsafe-perm --no-audit --no-fund /home/node/wrtc-0.4.7-dev.tgz

WORKDIR /opt/rapids/node

USER node
