ARG FROM_IMAGE
ARG DEVEL_IMAGE

FROM ${DEVEL_IMAGE} as devel

WORKDIR /home/node

RUN cp                                   \
    /opt/rapids/wrtc-0.4.7-dev.tgz       \
    /opt/rapids/rapidsai-core-*.tgz      \
    /opt/rapids/rapidsai-cuda-*.tgz      \
    /opt/rapids/rapidsai-glfw-*.tgz      \
    /opt/rapids/rapidsai-webgl-*.tgz     \
    /opt/rapids/rapidsai-rmm-*.tgz       \
    /opt/rapids/rapidsai-cudf-*.tgz      \
    /opt/rapids/rapidsai-sql-*.tgz       \
    /opt/rapids/rapidsai-cuml-*.tgz      \
    /opt/rapids/rapidsai-cugraph-*.tgz   \
    /opt/rapids/rapidsai-cuspatial-*.tgz \
    /opt/rapids/rapidsai-io-*.tgz        \
    /opt/rapids/rapidsai-deck.gl-*.tgz   \
    /opt/rapids/rapidsai-jsdom-*.tgz     \
    /opt/rapids/rapidsai-demo-*.tgz      \
    . \
 && npm install --production --omit dev --omit peer --omit optional --legacy-peer-deps --force *.tgz

FROM ${FROM_IMAGE}

SHELL ["/bin/bash", "-c"]

USER root

# Install UCX
COPY --from=devel /usr/local/bin/ucx_info         /usr/local/bin/
COPY --from=devel /usr/local/bin/ucx_perftest     /usr/local/bin/
COPY --from=devel /usr/local/bin/ucx_read_profile /usr/local/bin/
COPY --from=devel /usr/local/include/ucm          /usr/local/include/
COPY --from=devel /usr/local/include/ucp          /usr/local/include/
COPY --from=devel /usr/local/include/ucs          /usr/local/include/
COPY --from=devel /usr/local/include/uct          /usr/local/include/
COPY --from=devel /usr/local/lib/libucm.a         /usr/local/lib/
COPY --from=devel /usr/local/lib/libucm.la        /usr/local/lib/
COPY --from=devel /usr/local/lib/libucm.so.0.0.0  /usr/local/lib/
COPY --from=devel /usr/local/lib/libucp.a         /usr/local/lib/
COPY --from=devel /usr/local/lib/libucp.la        /usr/local/lib/
COPY --from=devel /usr/local/lib/libucp.so.0.0.0  /usr/local/lib/
COPY --from=devel /usr/local/lib/libucs.a         /usr/local/lib/
COPY --from=devel /usr/local/lib/libucs.la        /usr/local/lib/
COPY --from=devel /usr/local/lib/libucs.so.0.0.0  /usr/local/lib/
COPY --from=devel /usr/local/lib/libuct.a         /usr/local/lib/
COPY --from=devel /usr/local/lib/libuct.la        /usr/local/lib/
COPY --from=devel /usr/local/lib/libuct.so.0.0.0  /usr/local/lib/
COPY --from=devel /usr/local/lib/pkgconfig        /usr/local/lib/
COPY --from=devel /usr/local/lib/ucx              /usr/local/lib/

RUN cd /usr/local/lib \
 && ln -s libucm.so.0.0.0 libucm.so \
 && ln -s libucp.so.0.0.0 libucp.so \
 && ln -s libucs.so.0.0.0 libucs.so \
 && ln -s libuct.so.0.0.0 libuct.so \
 \
 # Install dependencies
 && export DEBIAN_FRONTEND=noninteractive \
 && apt update \
 && apt install -y --no-install-recommends \
    # cuSpatial dependencies
    libgdal-dev \
    # X11 dependencies
    libxrandr2 libxinerama1 libxcursor1 \
    # Wayland dependencies
    wayland-protocols \
    libwayland-{bin,egl1,cursor0,client0,server0} \
    libxkbcommon0 libxkbcommon-x11-0 \
    # GLEW dependencies
    libglvnd0 libgl1 libglx0 libegl1 libgles2 libglu1-mesa \
    # UCX runtime dependencies
    libibverbs1 librdmacm1 libnuma1 \
    # node-canvas dependencies
    libcairo2 libpango-1.0-0 libpangocairo-1.0-0 libjpeg8 libgif7 librsvg2-2 \
    # SQL dependencies
    openjdk-8-jre-headless libboost-regex-dev libboost-system-dev libboost-filesystem-dev \
 # Clean up
 && apt autoremove -y && apt clean \
 && rm -rf \
    /tmp/* \
    /var/tmp/* \
    /var/lib/apt/lists/* \
    /var/cache/apt/archives/*

COPY --from=devel --chown=node:node /home/node/node_modules/ /home/node/node_modules/

USER node

WORKDIR /home/node

SHELL ["/bin/bash", "-l"]

CMD ["node"]
