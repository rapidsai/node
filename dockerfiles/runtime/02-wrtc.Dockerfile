ARG FROM_IMAGE
ARG DEVEL_IMAGE

FROM ${DEVEL_IMAGE} as devel

WORKDIR /home/node

COPY --chown=node:node .npmrc /home/node/.npmrc

RUN set -x \
 && echo -e "build env:\n$(env)" \
 && echo -e "build context:\n$(find .)" \
 && npm install --production --omit dev --omit peer --omit optional --legacy-peer-deps --force *.tgz \
 && npm dedupe  --production --omit dev --omit peer --omit optional --legacy-peer-deps --force

FROM ${FROM_IMAGE}

SHELL ["/bin/bash", "-c"]

USER root

RUN export DEBIAN_FRONTEND=noninteractive \
 && apt update --fix-missing \
 && apt install -y --no-install-recommends \
    libxi-dev libxrandr-dev \
 # Clean up
 && apt autoremove -y && apt clean \
 && rm -rf \
    /tmp/* \
    /var/tmp/* \
    /var/lib/apt/lists/*

COPY --from=devel --chown=node:node /home/node/node_modules /home/node/node_modules

USER node

WORKDIR /home/node

SHELL ["/bin/bash", "-l"]

CMD ["node"]
