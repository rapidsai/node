ARG FROM_IMAGE
ARG BUILD_IMAGE
ARG DEVEL_IMAGE

FROM ${BUILD_IMAGE} as build
FROM ${DEVEL_IMAGE} as devel

WORKDIR /home/node

RUN --mount=type=bind,from=build,source=/opt/rapids/,target=/tmp/rapids/ \
    npm install --omit=dev --omit=peer --omit=optional --legacy-peer-deps --force \
       /tmp/rapids/rapidsai-core-*.tgz    \
       /tmp/rapids/rapidsai-cuda-*.tgz    \
       /tmp/rapids/rapidsai-rmm-*.tgz     \
       /tmp/rapids/rapidsai-cudf-*.tgz    \
       /tmp/rapids/rapidsai-cugraph-*.tgz ;

FROM ${FROM_IMAGE}

SHELL ["/bin/bash", "-c"]

WORKDIR /home/node

COPY --from=devel --chown=node:node /home/node/node_modules .

SHELL ["/bin/bash", "-l"]

CMD ["node"]
