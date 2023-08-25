# Developing on bare-metal

This document describes how to build and test on a properly configured Ubuntu installation outside docker.

Note: Due to the complexity of installing, updating, and managing native dependencies, we recommend [using the devel containers](https://github.com/rapidsai/node/blob/main/DEVELOP.md) for day-to-day development.

## Quick links

* [Common tools and dependencies](#common-tools-and-dependencies)
  * [Additional per-module dependencies](#additional-per-module-dependencies)
* [Command to install most native dependencies](#command-to-install-most-native-dependencies)
* [Troubleshooting](#troubleshooting)

## Common tools and dependencies

The following dependencies are necessary to build any of the `node-rapids` native modules:

* [CUDA Toolkit v11.0+ and compatible driver](https://developer.nvidia.com/cuda-downloads).
* [node, npm](https://github.com/nvm-sh/nvm#installing-and-updating), and [yarn](https://yarnpkg.com/getting-started/install).
* [CMake v3.20.2+](https://cmake.org/) (recommend either the [apt repository](https://apt.kitware.com/) or self-installing shell script).
* `gcc-9` toolchain (available in Ubuntu via the official toolchain PPA `ppa:ubuntu-toolchain-r/test`)
* ```txt
  ninja-build sccache jq zlib1g-dev liblz4-dev clang-format-17 clangd-17 lldb-17
  ```

### Additional per-module dependencies

* `@rapidsai/cuspatial`

  ```txt
  libgdal-dev
  ```

* `@rapidsai/sql`

  ```txt
  maven openjdk-8-jdk libboost-regex-dev libboost-system-dev libboost-filesystem-dev
  ```

  (`openjdk-11-jdk` also acceptable)

  * [UCX v1.11.x](https://github.com/openucx/ucx.git)

    ```txt
    libibverbs-dev librdmacm-dev libnuma-dev libhwloc-dev
    ```

* `node-canvas`, `@rapidsai/glfw`, `@rapidsai/webgl`

  ```txt
  libxi-dev libxrandr-dev libxinerama-dev libxcursor-dev libcairo2-dev libpango1.0-dev libjpeg-dev libgif-dev librsvg2-dev extra-cmake-modules libwayland-dev wayland-protocols libxkbcommon-dev build-essential libxmu-dev libxi-dev libgl1-mesa-dev libegl1-mesa-dev libglu1-mesa-dev
  ```

## Command to install most native dependencies

We include a one-shot command for installing most C++ dependencies (in Ubuntu):

```bash
# Bootstrap a new dev environment -- only necessary to run once.
# Installs VSCode, C++ intellisense plugins, and system libraries.
# Checks whether individual components are already installed,
# and asks permission before installing new components.
yarn dev:install-cpp-dependencies
```

This script does not install GCC or the SQL module's dependencies. You should to install and manage those separately (via [`update-alternatives`](http://manpages.ubuntu.com/manpages/trusty/man8/update-alternatives.8.html) or similar). See [dev/dockerfiles/devel/main.Dockerfile](https://github.com/rapidsai/node/blob/main/dev/dockerfiles/devel/main.Dockerfile) for an example of installing gcc-9, the RDMA/Infiniband drivers, and building UCX.

## Troubleshooting

Some rememedies for potential error messages you may encounter.

* > unsupported GNU version! gcc versions later than 8 are not supported!

  Install a [compatible CUDA host compiler](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#system-requirements) for your CUDA toolkit and OS versions.

* >  No CMAKE_CUDA_COMPILER could be found.

  This likely means your CUDA toolkit bin directory isn't in your environment's `$PATH`.
  Run the following commands to append the CUDA toolkit bin directory to your path,
  then reinitialize your current shell environment:

  ```bash
  echo '
  export CUDA_HOME="/usr/local/cuda"
  export PATH="$PATH:$CUDA_HOME/bin"
  ' >> ~/.bashrc

  source ~/.bashrc
  ```

* > ninja: error: loading 'build.ninja': No such file or directory

  This means the CMake "configure" step did not succeed. You need to execute `yarn rebuild` from the top of the repo (to rebuild everything), or in the specific module that failed to configure.
