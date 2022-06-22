# Using the `node-rapids` docker images

## Quick links

* [About our docker images](#develop-with-docker-and-the-vscode-remote-containers-extension)
* [Retrieving the runtime images](#retrieving-the-runtime-images)
* [Running code in the runtime images](#running-code-in-the-runtime-images)
* [Extracting the packaged artifacts](#extracting-the-packaged-artifacts)

## About our docker images

We publish standalone docker images [to the `rapidsai/node` repository](https://github.com/orgs/rapidsai/packages/container/package/node) to the GitHub container registry on each PR merge.

These images are based on the [`nvidia/cuda`](https://hub.docker.com/r/nvidia/cuda) runtime image. They are intended to be used directly, extended with app code, or to serve as examples for building your own deployment images. The sources for each image are in [dev/dockerfiles/runtime](https://github.com/rapidsai/node/tree/main/dev/dockerfiles/runtime).

Our tag template scheme is as follows:

```txt
ghcr.io/rapidsai/node:{{ RAPIDS_VERSION }}-runtime-node{{ NODE_VERSION }}-cuda{{ CUDA_VERSION }}-ubuntu{{ UBUNTU_VERSION }}-{{ LIBRARY_NAME }}
```

The latest manifest is available at the [GitHub container registry page](https://github.com/orgs/rapidsai/packages/container/package/node).

## Retrieving the runtime images

The following will retrieve the docker image with each library (+ its native and JS dependencies) installed into the image's `/home/node/node_modules`:

```bash
REPO=ghcr.io/rapidsai/node

VERSIONS="22.06.00-runtime-node18.2.0-cuda11.6.2-ubuntu20.04"
docker pull $REPO:$VERSIONS-cudf
docker pull $REPO:$VERSIONS-cuml
docker pull $REPO:$VERSIONS-cugraph
docker pull $REPO:$VERSIONS-cuspatial

VERSIONS="22.06.00-runtime-node18.2.0-cuda11.6.2-ubuntu20.04"
docker pull $REPO:$VERSIONS-glfw

# Includes all the above RAPIDS libraries in a single image
docker pull $REPO:$VERSIONS-main

# Includes all the above RAPIDS libraries + demos in a single image
docker pull $REPO:$VERSIONS-demo
```

## Running code in the runtime images

Like the official node images, the default command in the runtime images is `node`. You can run the following test to validate the GPU is available to your containers:

```bash
REPO=ghcr.io/rapidsai/node
VERSIONS="22.06.00-runtime-node18.2.0-cuda11.6.2-ubuntu20.04"

# Be sure to pass either the `--runtime=nvidia` or `--gpus` flag!
docker run --rm --gpus=0 $REPO:$VERSIONS-cudf \
    -p "const {Series, DataFrame} = require('@rapidsai/cudf');\
        new DataFrame({ a: Series.new([0, 1, 2]) }).toString()"
>   a
> 0.0
> 1.0
> 2.0

```

You can mount your host's X11 socket and `$DISPLAY` envvar, then launch demos that render via OpenGL:

```bash
REPO=ghcr.io/rapidsai/node
VERSIONS="22.06.00-runtime-node18.2.0-cuda11.6.2-ubuntu20.04"

# Be sure to pass either the `--runtime=nvidia` or `--gpus` flag!
docker run --rm \
    --runtime=nvidia \
    -e "DISPLAY=$DISPLAY" \
    -v "/etc/fonts:/etc/fonts:ro" \
    -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    -v "/usr/share/fonts:/usr/share/fonts:ro" \
    -v "/usr/share/icons:/usr/share/icons:ro" \
    $REPO:$VERSIONS-demo \
    npx @rapidsai/demo-graph
```

<details>
<summary>Click here to see a demo of the above command:</summary>
<img src="docs/images/docker-x11-socket-forwarding.gif"/>
</details>

## Extracting the packaged artifacts

We also publish the container of npm-packed `.tgz` artifacts installed into each runtime image.

You can use the following technique to install the npm-packed modules into another container or bare-metal cloud instance, provided you also install any runtime dependencies (a gcc-9-compatible `libstdc++`, matching CUDA Toolkit, etc.) needed by the module.

```bash
REPO=ghcr.io/rapidsai/node
VERSIONS="22.06.00-devel-node18.2.0-cuda11.6.2-ubuntu20.04"

# Pull the latest image of the packaged .tgz artifacts
docker pull $REPO:$VERSIONS-packages

# Copy the packaged .tgz artifacts, including any of its `@rapidsai/*` dependencies
docker run --rm -v "$PWD:/out" \
    $REPO:$VERSIONS-packages \
    bash -c "cp \
        /opt/rapids/rapidsai-core-*.tgz \
        /opt/rapids/rapidsai-cuda-*.tgz \
        /opt/rapids/rapidsai-rmm-*.tgz  \
        /opt/rapids/rapidsai-cudf-*.tgz \
        /out/"

# Install the npm-packed .tgz (+ npm dependencies) into `./node_modules`
npm install --force --production --legacy-peer-deps \
    --omit dev --omit peer --omit optional \
    *.tgz

```
