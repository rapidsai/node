ARG FROM_IMAGE
ARG DEVEL_IMAGE

FROM ${DEVEL_IMAGE} as devel

WORKDIR /home/node

RUN cp                                         \
    /opt/node-rapids/.npmrc                    \
    /opt/node-rapids/build/rapidsai-core-*.tgz \
    /opt/node-rapids/build/nvidia-cuda-*.tgz   \
    /opt/node-rapids/build/rapidsai-rmm-*.tgz  \
    /opt/node-rapids/build/rapidsai-cudf-*.tgz \
    /opt/node-rapids/build/rapidsai-cuml-*.tgz \
    . \
 && npm install --production --omit dev --omit peer --omit optional --legacy-peer-deps --force *.tgz \
 && npm dedupe  --production --omit dev --omit peer --omit optional --legacy-peer-deps --force


FROM ${FROM_IMAGE}

SHELL ["/bin/bash", "-c"]

ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

COPY --from=devel --chown=node:node /home/node/node_modules/ /home/node/node_modules/

WORKDIR /home/node

SHELL ["/bin/bash", "-l"]

CMD ["node"]
