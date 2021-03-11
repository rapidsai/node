# <div align="left"><img src="https://rapids.ai/assets/images/rapids_logo.png" width="90px"/>&nbsp; rapids-js


The `rapids-js` project comprises of collection of JavaScript bindings to the following Rapids libraries:
1. [cudf](https://github.com/rapidsai/cudf)
2. [cugraph](https://github.com/rapidsai/cugraph)
3. [cuspatial](https://github.com/rapidsai/cuspatial)
4. [rmm](https://github.com/rapidsai/rmm)

Additionally, `rapids-js` includes JavaScript bindings to [cuda](https://developer.nvidia.com/cuda-toolkit) and [glfw](https://github.com/glfw/glfw).

For detailed APIs, follow our [API Documentation](https://rapidsai.github.io/rapids-js/).

## Setup & Installation

#### CUDA/GPU requirements
- CUDA 10.1+
- NVIDIA driver 418.39+
- Pascal architecture or better (Compute Capability >=6.0)


The above mentioned js-modules can be installed separately as per the use cases. To get started with `rapids-js`, follow the [Setup Instructions](docs/setup.md).

## Demos

The demos module contains a bunch of examples which use a combination of rapids-js modules to re-implement some browser+webgl based examples. Some of them include:

- [deck](modules/demo/deck/)
- [graph](modules/demo/graph/)
- [luma](modules/demo/luma/)
- [tfjs](modules/demo/tfjs/)
- [umap](modules/demo/umap/)
- [xterm](modules/demo/xterm/)

After you build the modules, just run `yarn demo` from the command line, and choose the demo you want to run.


## License

This work is licensed under the [Apache-2.0 license](./LICENSE).

---
