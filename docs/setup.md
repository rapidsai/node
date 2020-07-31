

## Setup

### Dependencies


#### With Docker 

You can build inside and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) container. Once you have followed the setup instructions there, issue the following commands:

```bash
# Build the development container
docker-compose build devel

# Start the development container
docker-compose run --rm devel
```

If you would like to run a parallel build for the container, you can issue this `build` command, instead of the one above:
```bash
PARALLEL_LEVEL=$(nproc) docker-compose build devel
```

#### Without Docker

Alternatively, you can build directly on a Ubuntu system without Docker. To install the necessary dependencies, issue the following commands:

```bash
# Bootstrap a new dev environment (only necessary to run once). Installs tools like
# VSCode, C++ plugins, and system libraries and dependencies (say "yes" to prompts)
npm run dev:install-cpp-dependencies

# Boostrap the yarn workspaces and install dependencies
yarn
```

### Building


To build the C++ and Typescript, issue the command:

```bash
npm run build
```

To perform a clean build, issue the command:

```bash
npm run build -- --clean
```

### Running

You can run a demo to test the build by issuing the command:

```bash
npm run demo modules/demo/luma 01
```

### Troubleshooting

Some rememedies for potential error messages you may encounter.

* > unsupported GNU version! gcc versions later than 8 are not supported!

  Set the configured gcc/g++ versions with these commands:

  ```bash
  sudo update-alternatives --config gcc
  sudo update-alternatives --config g++
  ```

* >  missing clangd

   Congfigure clang 12 with this command:

  ```bash
  sudo update-alternatives --install /usr/bin/clangd clangd /usr/bin/clangd-12 100
  ```

* >  No CMAKE_CUDA_COMPILER could be found.

  Add this `~/.bashrc` and open a new terminal:

  ```bash
  export CUDA_HOME="/usr/local/cuda"
  export PATH="$PATH:$CUDA_HOME/bin"
  ```

