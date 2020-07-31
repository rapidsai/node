# Setup

## Quick links

* [Develop with docker](#develop-with-docker)
  * [Installation](#installation)
  * [Usage](#usage)
* [Develop without docker](#develop-without-docker)
  * [Dependencies](#dependencies)
* [Setup yarn workspaces](#setup-yarn-workspaces)
* [Build the node modules](#build-the-node-modules)
* [Running the demos](#tunning-the-demos)
* [Troubleshooting](#troubleshooting)

## Develop with docker

You can build and run the C++ and TypeScript in a [docker](https://docker.com/) container with the [nvidia-container-runtime](https://github.com/NVIDIA/nvidia-docker) and [docker-compose](https://github.com/docker/compose/). This approach helps eliminate potential incompatibilities between the tools and environment in your current OS vs. the tools needed to build the native C++ and CUDA node modules.

### Installation

See our [`docker`, `docker-compose`, and `nvidia-container-runtime` installation and setup instructions](docker/installation.md).

### Usage

To build and run the development container, issue the following commands:

```bash
# Build the development container
docker-compose build devel

# Start the development container
docker-compose run --rm devel

# Build and run demos from inside the container (more info below)
\# npm run build
\# npm run demo modules/demo/luma 01
```

## Develop without docker

You can also build and run on a properly configured Linux installation without docker.

### Dependencies

We assume you have [node, npm](https://github.com/nvm-sh/nvm#installing-and-updating), [yarn](https://yarnpkg.com/getting-started/install), and [CMake](https://cmake.org/) installed.

<details>
<summary>Click here to see Ubuntu 16.04+ CMake installation commands:</summary>
<pre>
# Install CMake v3.17.4, or select any CMake 3.17.x release in https://github.com/Kitware/CMake/releases
CMAKE_VERSION=3.17.4<br/>
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
npm run dev:install-cpp-dependencies
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
# Run CMake configuration, find/build native dependencies, and compile C++ and TypeScript
npm run build
# Perform a fast recompile (without the CMake configuration step):
npm run build -- --fast
# Perform a clean build:
npm run build -- --clean
```

## Running the demos

You can run a demo to test the build by issuing the command:

```bash
npm run demo modules/demo/luma 01
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
