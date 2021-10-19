# <div align="left"><img src="https://rapids.ai/assets/images/rapids_logo.png" width="90px"/>&nbsp; node-rapids

[`node-rapids`](https://github.com/rapidsai/node) is a collection of Node.js native addons for the [NVIDIA RAPIDS](https://rapids.ai/) suite of GPU-accelerated data-science and ETL libraries on Linux and WSL2.

* [`@rapidsai/rmm`](https://github.com/rapidsai/node/tree/main/modules/rmm) - [RAPIDS Memory Manager](https://github.com/rapidsai/rmm)
* [`@rapidsai/cudf`](https://github.com/rapidsai/node/tree/main/modules/cudf) - [RAPIDS GPU DataFrame](https://github.com/rapidsai/cudf)
* [`@rapidsai/cuml`](https://github.com/rapidsai/node/tree/main/modules/cuml) - [RAPIDS Machine Learning Library](https://github.com/rapidsai/cuml)
* [`@rapidsai/cugraph`](https://github.com/rapidsai/node/tree/main/modules/cugraph) - [RAPIDS Graph Analytics Library](https://github.com/rapidsai/cugraph)
* [`@rapidsai/cuspatial`](https://github.com/rapidsai/node/tree/main/modules/cuspatial) - [RAPIDS Accelerated GIS Library](https://github.com/rapidsai/cuspatial)
* [`@rapidsai/sql`](https://github.com/rapidsai/node/tree/main/modules/sql) - Multi-node/multi-GPU accelerated SQL execution engine

`node-rapids` also includes a limited set of bindings to other necessary native APIs:

* [`@rapidsai/cuda`](https://github.com/rapidsai/node/tree/main/modules/cuda) - Interact with GPUs via the [CUDA Runtime APIs](https://developer.nvidia.com/cuda-toolkit)
* [`@rapidsai/glfw`](https://github.com/rapidsai/node/tree/main/modules/glfw) - Create platform-agnostic native windows with OpenGL contexts via [GLFW](https://github.com/glfw/glfw)
* [`@rapidsai/webgl`](https://github.com/rapidsai/node/tree/main/modules/webgl) - Provides a [`WebGL2RenderingContext`](https://developer.mozilla.org/en-US/docs/Web/API/WebGL2RenderingContext) via [OpenGL ES](https://www.khronos.org/opengles)

`node-rapids` uses the ABI-stable [`N-API`](https://nodejs.org/api/n-api.html) via [`node-addon-api`](https://github.com/nodejs/node-addon-api), so the same binaries work in both node and Electron.

See the [API docs](https://rapidsai.github.io/node/) for detailed information about each module.

## Getting started

Due to the complexity of packaging and managing native dependencies, pre-packaged builds of the `node-rapids` modules are available via our [public docker images](https://github.com/orgs/rapidsai/packages/container/package/node).

See our docs on [using the public `node-rapids` docker images](https://github.com/rapidsai/node/tree/main/USAGE.md).

## Getting involved

See [DEVELOP.md](https://github.com/rapidsai/node/blob/main/DEVELOP.md) for details on setting up a local dev environment and building the code.

## Tracking Progress

You can review [BINDINGS.md](https://github.com/rapidsai/node/blob/main/BINDINGS.md) to see which bindings have been completed for each of the RAPIDS libraries.

## Demos

The demos module contains a bunch of examples which use a combination of node-rapids modules to re-implement some browser+webgl based examples. Some of them include:

* [luma.gl](https://github.com/rapidsai/node/tree/main/modules/demo/luma/)
* [deck.gl](https://github.com/rapidsai/node/tree/main/modules/demo/deck/)
* [XTerm.js](https://github.com/rapidsai/node/tree/main/modules/demo/xterm/)
* [forceAtlas2](https://github.com/rapidsai/node/tree/main/modules/demo/graph/)
* [TensorFlow.js](https://github.com/rapidsai/node/tree/main/modules/demo/tfjs/)

After you build the modules, run `yarn demo` from the command line to choose the demo you want to run.

## License

This work is licensed under the [Apache-2.0 license](https://github.com/rapidsai/node/tree/main/LICENSE).

---
