ARG FROM_IMAGE
ARG DEVEL_IMAGE

FROM ${DEVEL_IMAGE} as devel

WORKDIR /home/node

RUN cp                               \
    /opt/rapids/wrtc-0.4.7-dev.tgz   \
    /opt/rapids/rapidsai-core-*.tgz  \
    /opt/rapids/rapidsai-glfw-*.tgz  \
    /opt/rapids/rapidsai-webgl-*.tgz \
    . \
 && npm install --production --omit dev --omit peer --omit optional --legacy-peer-deps --force *.tgz

FROM ${FROM_IMAGE}

SHELL ["/bin/bash", "-c"]

USER root

RUN export DEBIAN_FRONTEND=noninteractive \
 && apt update \
 && apt install -y --no-install-recommends \
    # X11 dependencies
    libxrandr2 libxinerama1 libxcursor1 \
    # Wayland dependencies
    libwayland-bin \
    wayland-protocols \
    libwayland-server0 \
    libwayland-egl1 libwayland-egl++0 \
    libwayland-cursor0 libwayland-cursor++0 \
    libwayland-client0 libwayland-client++0 libwayland-client-extra++0 \
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

COPY --from=devel --chown=node:node /home/node/node_modules/ /home/node/node_modules/

USER node

WORKDIR /home/node

SHELL ["/bin/bash", "-l"]

CMD ["node"]
