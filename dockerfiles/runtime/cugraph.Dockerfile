ARG FROM_IMAGE
ARG DEVEL_IMAGE

FROM ${DEVEL_IMAGE} as devel

WORKDIR /home/node

RUN cp                                 \
    /opt/rapids/rapidsai-core-*.tgz    \
    /opt/rapids/rapidsai-cuda-*.tgz    \
    /opt/rapids/rapidsai-rmm-*.tgz     \
    /opt/rapids/rapidsai-cudf-*.tgz    \
    /opt/rapids/rapidsai-cugraph-*.tgz \
    . \
 && npm install --production --omit dev --omit peer --omit optional --legacy-peer-deps --force *.tgz

FROM ${FROM_IMAGE}

SHELL ["/bin/bash", "-c"]

ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

COPY --from=devel --chown=node:node /home/node/node_modules/ /home/node/node_modules/

WORKDIR /home/node

SHELL ["/bin/bash", "-l"]

CMD ["node"]
