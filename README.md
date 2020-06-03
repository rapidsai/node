## rapids-js

> 

### Setup
```bash
# Bootstrap a new development environment, only necessary to run once.
# Installs dev tools like VSCode, C++ plugins, and system libraries and dependencies
npm run dev:install-cpp-dependencies
# Boostrap the yarn workspaces and install dependencies
yarn
# Build all the C++ and TypeScript
npm run build
# Run a demo
npm run demo modules/demo/luma 01
```

#### With Docker (nvidia-docker + docker-compose)
```bash
# Build the development container
docker-compose build devel
# Start the development container
docker-compose run --rm devel
# Build the C++ and TypeScript in the development container
\# npm run build
# Run a demo in the development container
\# npm run demo modules/demo/luma 01
```

### Apache-2.0

This work is licensed under the [Apache-2.0 license](./LICENSE).

---
