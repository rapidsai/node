# <div align="left"><img src="https://rapids.ai/assets/images/rapids_logo.png" width="90px"/>&nbsp; node-rapids

[`node-rapids`](https://github.com/rapidsai/node) is collection of `node` native C++ addons for the RAPIDS libraries:

* [`@rapidsai/rmm`](https://github.com/rapidsai/node/tree/main/modules/rmm) - Bindings to the [RAPIDS Memory Manager](https://github.com/rapidsai/rmm)
* [`@rapidsai/cudf`](https://github.com/rapidsai/node/tree/main/modules/cudf) - Bindings to the [RAPIDS GPU DataFrame](https://github.com/rapidsai/cudf)
* [`@rapidsai/cugraph`](https://github.com/rapidsai/node/tree/main/modules/cugraph) - Bindings to the [RAPIDS Graph Analytics Library](https://github.com/rapidsai/cugraph)
* [`@rapidsai/cuspatial`](https://github.com/rapidsai/node/tree/main/modules/cuspatial) - Bindings to the [RAPIDS Accelerated GIS Library](https://github.com/rapidsai/cuspatial)

Additionally, `node-rapids` includes _limited_ bindings to:

* [`@nvidia/cuda`](https://github.com/rapidsai/node/tree/main/modules/cuda) - Interact with GPUs via the [CUDA Runtime APIs](https://developer.nvidia.com/cuda-toolkit)
* [`@nvidia/glfw`](https://github.com/rapidsai/node/tree/main/modules/glfw) - Create platform-agnostic native windows with OpenGL contexts via [GLFW](https://github.com/glfw/glfw)
* [`@nvidia/webgl`](https://github.com/rapidsai/node/tree/main/modules/webgl) - Provides a [`WebGL2RenderingContext`](https://developer.mozilla.org/en-US/docs/Web/API/WebGL2RenderingContext) via [OpenGL ES](https://www.khronos.org/opengles)
* [`@rapidsai/sql`](https://github.com/rapidsai/node/tree/main/modules/sql) - GPU-accelerated SQL queries

See the [API docs](https://rapidsai.github.io/node/) for detailed information about each module.

## Setup

#### System/CUDA/GPU requirements

- Ubuntu 16.04+ (other Linuxes may work, but untested)
- Docker 19.03+ (optional)
- docker-compose v1.28.5+ (optional)
- CUDAToolkit 10.1+
- NVIDIA driver 418.39+
- Pascal architecture (Compute Capability >=6.0) or better

To get started building and using `node-rapids`, follow the [setup instructions](https://github.com/rapidsai/node/tree/main/docs/setup.md).

The `node-rapids` modules are not yet available on npm. They must be built locally or in our Docker environments.

## Notebooks

We've included a container for launching [`nteract/desktop`](https://nteract.io/desktop) with access to locally built `node-rapids` modules:

```shell
# Make a local .env file of var overrides for the containers and build
cp .env.sample .env

# Build the development and nteract containers (only necessary once)
yarn docker:build:devel && yarn docker:build:devel:notebook

# Compile the TypeScript and C++ modules inside the development container
yarn docker:run:devel bash -c 'yarn && yarn build'

# Start a containerized nteract/desktop with the source tree as Docker volume mounts
yarn docker:run:devel:notebook
```

`node-rapids` packages are built with [`N-API`](https://nodejs.org/api/n-api.html) via the [`node-addon-api`](https://github.com/nodejs/node-addon-api) library.


## Demos

The demos module contains a bunch of examples which use a combination of node-rapids modules to re-implement some browser+webgl based examples. Some of them include:

- [deck.gl](https://github.com/rapidsai/node/tree/main/modules/demo/deck/)
- [luma.gl](https://github.com/rapidsai/node/tree/main/modules/demo/luma/)
- [TensorFlow.js](https://github.com/rapidsai/node/tree/main/modules/demo/tfjs/)
- [XTerm.js](https://github.com/rapidsai/node/tree/main/modules/demo/xterm/)
- [CUML UMAP](https://github.com/rapidsai/node/tree/main/modules/demo/ipc/umap/)

After you build the modules, run `yarn demo` from the command line to choose the demo you want to run.

## Bindings Progress

You can review [BINDINGS.md](https://github.com/rapidsai/node/blob/main/BINDINGS.md) to see which bindings have been completed for each of the RAPIDS libraries.

## FAQs

### What is the `.env` file used for and what variables can be set in it?

The `.env` file is used by the [docker-compose.*.yml](https://github.com/rapidsai/node/blob/main/docker-compose.devel.yml) files when building and running the containers (i.e. `yarn docker:{build,run}:{devel,runtime}`). Any variables in the [docker-compose.*.yml](https://github.com/rapidsai/node/blob/main/docker-compose.devel.yml) files can be set in the `.env` file. The [.env.sample](https://github.com/rapidsai/node/blob/main/.env.sample) file includes some common variables that may be set in the `.env` file.

## License

This work is licensed under the [Apache-2.0 license](https://github.com/rapidsai/node/tree/main/LICENSE).

---
