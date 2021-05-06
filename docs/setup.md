# Setup

## Quick links

* [Develop with docker](#develop-with-docker)
  * [Installation](#installation)
  * [Usage](#usage)
* [Develop without docker](#develop-without-docker)
  * [Dependencies](#dependencies)
* [Setup yarn workspaces](#setup-yarn-workspaces)
* [Build the node modules](#build-the-node-modules)
* [Running the demos](#running-the-demos)
* [Troubleshooting](#troubleshooting)

## Develop with docker

You can build and run the C++ and TypeScript in a [docker](https://docker.com/) container with the [nvidia-container-runtime](https://github.com/NVIDIA/nvidia-docker) and [docker-compose](https://github.com/docker/compose/). This approach helps eliminate potential incompatibilities between the tools and environment in your current OS vs. the tools needed to build the native C++ and CUDA node modules.

### Installation

See our [`docker`, `docker-compose`, and `nvidia-container-runtime` installation and setup instructions](docker/installation.md).

### Usage

To build and run the development container, issue the following commands:

```bash
# Build the development container
yarn docker:build:devel

# Start the development container
yarn docker:run:devel
```

Now execute the following commands inside this running container:

* [Setup yarn workspaces](#setup-yarn-workspaces)
* [Build the node modules](#build-the-node-modules)
* [Running the demos](#running-the-demos)

## Develop without docker

You can also build and run on a properly configured Linux installation without docker.

### Dependencies

We assume you have [node, npm](https://github.com/nvm-sh/nvm#installing-and-updating), [yarn](https://yarnpkg.com/getting-started/install), [CMake v3.20.2+](https://cmake.org/), and [CUDA Toolkit 11.0+](https://developer.nvidia.com/cuda-downloads) installed.

<details>
<summary>Click here to see Ubuntu 16.04+ CMake installation commands:</summary>
<pre>
# Install CMake v3.20.2, or select any CMake 3.18.x release in https://github.com/Kitware/CMake/releases
CMAKE_VERSION=3.20.2<br/>
wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}.tar.gz \
 && tar -xvzf cmake-${CMAKE_VERSION}.tar.gz && cd cmake-${CMAKE_VERSION} \
 && ./bootstrap --system-curl --parallel=$(nproc) && sudo make install -j \
 && cd - && rm -rf ./cmake-${CMAKE_VERSION} ./cmake-${CMAKE_VERSION}.tar.gz
</pre>
</details>

To install the rest of the dependencies necessary for building the native C++ and CUDA node modules, issue the following command:

```bash
# Bootstrap a new dev environment -- only necessary to run once.
# Installs VSCode, C++ intellisense plugins, and system libraries.
# Checks whether individual components are already installed,
# and asks permission before installing new components.
yarn dev:install-cpp-dependencies
```

## Setup yarn workspaces

Install and boostrap the monorepo's yarn workspaces:

```bash
# Boostrap the yarn workspaces and install node module development dependencies
yarn
```

## Build the node modules

To build the C++ and Typescript, issue any of the following commands:

```bash
# Run CMake configure, find native dependencies, and compile C++/TypeScript
yarn build
# Perform a fast recompile (without the CMake configure step):
yarn compile
# Perform a clean reconfigure/rebuild:
yarn rebuild
```

### C++-only builds

Issue the above commands with the `cpp:` prefix to only build the C++ modules. The `:debug` suffix will run each command with `CMAKE_BUILD_TYPE=Debug`.

```bash
yarn cpp:build # or cpp:build:debug
yarn cpp:compile # or cpp:compile:debug
yarn cpp:rebuild # or cpp:rebuild:debug
```

These npm scripts are also available in each module. Running them from each module's directory will output GCC colors.

## Running the demos

You can run a demo to test the build by issuing the command:

```bash
yarn demo modules/demo/luma 01
```

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

  Need to execure `yarn rebuild` from the top from inside the module
