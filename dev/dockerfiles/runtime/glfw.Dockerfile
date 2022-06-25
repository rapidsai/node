ARG FROM_IMAGE
ARG BUILD_IMAGE
ARG DEVEL_IMAGE

FROM ${BUILD_IMAGE} as build
FROM ${DEVEL_IMAGE} as devel

WORKDIR /home/node

RUN --mount=type=bind,from=build,source=/opt/rapids/,target=/tmp/rapids/ \
    npm install --omit=dev --omit=peer --omit=optional --legacy-peer-deps --force \
       /tmp/rapids/wrtc-dev.tgz         \
       /tmp/rapids/rapidsai-core-*.tgz  \
       /tmp/rapids/rapidsai-glfw-*.tgz  \
       /tmp/rapids/rapidsai-webgl-*.tgz ;

FROM ${FROM_IMAGE}

SHELL ["/bin/bash", "-c"]

USER root

RUN export DEBIAN_FRONTEND=noninteractive \
 && apt update \
 && apt install -y --no-install-recommends \
    # X11 dependencies
    libxrandr2 libxinerama1 libxcursor1 \
    # Wayland dependencies
    wayland-protocols \
    libwayland-{bin,egl1,cursor0,client0,server0} \
    libxkbcommon0 libxkbcommon-x11-0 \
    # GLEW dependencies
    libglvnd0 libgl1 libglx0 libegl1 libgles2 libglu1-mesa \
    # node-canvas dependencies
    libcairo2 libpango-1.0-0 libpangocairo-1.0-0 libjpeg8 libgif7 librsvg2-2 \
 # Clean up
 && apt autoremove -y && apt clean \
 && rm -rf \
    /tmp/* \
    /var/tmp/* \
    /var/lib/apt/lists/* \
    /var/cache/apt/archives/*

USER node

WORKDIR /home/node

COPY --from=devel --chown=node:node /home/node/node_modules .

SHELL ["/bin/bash", "-l"]

CMD ["node"]
