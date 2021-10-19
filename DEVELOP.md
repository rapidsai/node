# Development

This document is intended for anyone who wants to clone and build the `node-rapids` modules from source. If you're interested in using `node-rapids` packages in your app, see our docs on [using the public `node-rapids` docker images](https://github.com/rapidsai/node/tree/main/USAGE.md).

## Quick links

* [Develop with docker and VSCode Remote Containers](#develop-with-docker-and-the-vscode-remote-containers-extension)
* [Setup yarn workspace](#setup-yarn-workspace)
* [Build the modules](#build-the-modules)
  * [Build modules individually](#build-modules-individually)
* [Running the demos](#running-the-demos)
* [Running notebooks](#running-notebooks)
* [Troubleshooting](#troubleshooting)
* [Develop on bare-metal](#develop-on-bare-metal)

## System/GPU requirements

* Ubuntu 18.04+ (other Linuxes should work, but untested)
* [CUDA Toolkit 11.0+](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#system-requirements)
* NVIDIA driver 465+
* \>= Pascal architecture (Compute Capability >=7.0)
* `docker-ce` 19.03+ (optional)
* `docker-compose` v1.28.5+ (optional)

## Develop with docker and [VSCode Remote Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

You can build and run the C++ and TypeScript in a [docker](https://docker.com/) container with the [nvidia-container-runtime](https://github.com/NVIDIA/nvidia-docker) and [docker-compose](https://github.com/docker/compose/).

This eliminates potential incompatibilities between the tools and environment in your base OS vs. the tools needed to build, format, lint, debug, and provide intellisense for the C++ native modules.

### Install docker and VSCode extensions

See our `docker`, `docker-compose`, and `nvidia-container-runtime` [installation and setup instructions](https://github.com/rapidsai/node/tree/main/docs/docker/installation.md).

Install the [VSCode Remote Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).

We recommend installing the following extensions in the development container:

* [CMake](https://marketplace.visualstudio.com/items?itemName=twxs.cmake)
* [Microsoft C/C++ tools](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools)
* [Clang Format](https://marketplace.visualstudio.com/items?itemName=xaver.clang-format)
* [Clangd](https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.vscode-clangd)
* [Tasks Shell Input](https://marketplace.visualstudio.com/items?itemName=augustocdias.tasks-shell-input)
* [ESLint](https://marketplace.visualstudio.com/items?itemName=dbaeumer.vscode-eslint)

### Clone the repository

```bash
git clone https://github.com/rapidsai/node.git
```

### Create a `.env` file for local development

Create a local [`.env` file](https://docs.docker.com/compose/environment-variables/#the-env-file) for `docker-compose`. This file controls high-level features outside the container, like CUDA and Ubuntu version, and more fine-grained features inside the container like the user, number of parallel jobs, and compiler caching.

```bash
# Make a local .env file of envvar overrides for the container
cp .env.sample .env
# Modify the .env file to match your system's parameters
nano .env
```

### Starting and connecting to the development container

To pull and run the development container, issue the following commands:

```bash
# Pull the main development container built in node-rapids CI (fastest)
docker-compose -f docker-compose.devel.yml pull main

# Or build the development containers locally (e.g. after making changes)
yarn docker:build:devel:main

# Start the main development container
yarn docker:run:devel
```

After starting the container, you should see it in the VSCode Remote Containers extension.

<details>
<summary>Click the "Attach to Container" icon to open a new VSCode window in the context of the development container:</summary>
<img src="docs/images/vscode-attach-to-container.png"/>
</details>
<br/>

## Setup yarn workspace

Install and boostrap the monorepo's yarn workspace inside the development container:

```bash
# Boostrap the yarn workspace, set up monorepo symlinks, and install dev dependencies
yarn
```

## Build the modules

To build the TypeScript and C++ of each module:

```bash
yarn build
```

To build the TypeScript half of each module:

```bash
yarn tsc:build
```

To build the C++ half of each module:

```bash
yarn cpp:build
```

### Build modules individually

The npm scripts in the top-level `package.json` operate on every module (via [yarn workspaces](https://classic.yarnpkg.com/lang/en/docs/workspaces/) and [lerna](https://github.com/lerna/lerna)), but often it's easier to build a single module. Because each module also defines those scripts, we can use `yarn workspace` to run them in isolation:

```bash
# Run the `cpp:build` npm script in the cuDF module only
yarn workspace @rapidsai/cudf cpp:build
```

### Escape hatch

If something goes haywire in the dependency tree or build cache, we include an "escape hatch" command to clean and reinstall/rebuild all dependencies and build directories inside the container. Feel free to run this as often as you need:

```bash
yarn nuke:from:orbit
```

### CMake and Ninja

We use [cmake-js](https://www.npmjs.com/package/cmake-js) to generate the C++ build scripts and follow a similar naming convention in our npm scripts:

```txt
Usage: cmake-js [<command>] [options]

Commands:
  configure        Configure CMake project
  build            Build the project (will configure first if required)
  reconfigure      Clean the project directory then configure the project
  rebuild          Clean the project directory then build the project
  compile          Build the project, and if build fails, try a full rebuild
```

These `cmake-js` commands are exposed as npm scripts in the top-level `package.json`:

```txt
yarn cpp:configure        Configure CMake project
yarn cpp:build            Build the project (will configure first if required)
yarn cpp:reconfigure      Clean the project directory then configure the project
yarn cpp:rebuild          Clean the project directory then build the project
yarn cpp:compile          Build the project, and if build fails, try a full rebuild
```

Adding the `:debug` suffix to the above command will build the C++ projects (and their dependencies!) with `CMAKE_BUILD_TYPE=Debug`:

```txt
yarn cpp:rebuild:debug
```

This is potentially expensive but is required if you need to run in the debugger.

#### CMake

CMake is a "build script generator." It uses a `CMakeLists.txt` as a description of how to generate C++ build scripts.

A CMake build is separated into two phases; a "configuration" phase (run CMake to generate the build scripts), and a "compile" phase (executes commands in the generated build scripts to compile C++ source into object files).

Certain information about how to build C++ source into object files is determined in the "configuration" phase. If this information changes, the configuration phase must be re-run. For example, the C++ `.cpp` file paths are found in the configuration phase, so adding a new `.cpp` file requires regenerating (i.e. "reconfiguring") the build scripts.

#### Ninja

[Ninja](https://ninja-build.org/) is a build tool to compile C++ sources into object files. The CMake configuration phase generates ninja build scripts.

## Running the demos

The demos can be run via the top-level `yarn demo` npm script:

```bash
# Show a list of available demos
yarn demo

# Run a specific demo with extra arguments passed to the selected demo
yarn demo modules/demo/luma 01
```

## Running notebooks

We've included a container for launching [`nteract/desktop`](https://nteract.io/desktop) with access to locally built `node-rapids` modules:

```shell
# Compile the TypeScript and C++ modules inside the development container (only necessary if it hasn't already been built)
yarn docker:run:devel bash -c 'yarn && yarn build'

# Build the nteract notebook container (only necessary if it hasn't already been built)
yarn docker:build:devel:notebook

# Start a containerized `nteract/desktop` instance with the source tree as a docker volume mount
yarn docker:run:devel:notebook
```

## Troubleshooting

Potential error messages you may encounter while building:

* > ninja: error: loading 'build.ninja': No such file or directory

  This means the CMake "configure" step did not succeed. You need to execute `yarn rebuild` from the top of the repo (to rebuild everything), or in the specific module that failed to configure.

## Develop on bare-metal

Though not usually advised, follow [these instructions](https://github.com/rapidsai/node/tree/main/docs/develop-on-bare-metal.md) to develop outside docker on properly configured Ubuntu host.
