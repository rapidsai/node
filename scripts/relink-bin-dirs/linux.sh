#!/usr/bin/env bash

set -Eeo pipefail

DIR="$(pwd)"
BIN="$(realpath node_modules/.bin)"
DIRS=$(npx lerna exec --scope "@nvidia/*" "echo \$PWD")
for DIR in $DIRS; do
    mkdir -p "$DIR/node_modules"
    rm -rf "$DIR/node_modules/.bin"
    ln -sf "$BIN" "$DIR/node_modules/.bin"
done

NODE_BIN_PATH="$(which node)"
NAPI_INCLUDE_DIR="$PWD/node_modules/node-addon-api"
NODE_INCLUDE_DIR="${NODE_BIN_PATH%/bin/node}/include"
THRUST_INCLUDE_DIR="${CUDA_HOME:-/usr/local/cuda}/include/thrust"

for REPO in cuda glfw webgl; do
    DIR=$(npx lerna exec --scope "@nvidia/$REPO" "echo \$PWD" | head -n1)
    # symlink node headers
    ln -sf "$NODE_INCLUDE_DIR/node/node_api.h" "$DIR/include/node_api.h"
    # symlink napi headers
    ln -sf "$NAPI_INCLUDE_DIR/napi.h" "$DIR/include/napi.h"
    ln -sf "$NAPI_INCLUDE_DIR/napi-inl.h" "$DIR/include/napi-inl.h"
    ln -sf "$NAPI_INCLUDE_DIR/napi-inl.deprecated.h" "$DIR/include/napi-inl.deprecated.h"
done

# symlink thrust
NODE_CUDA_INCLUDE_DIR=$(npx lerna exec --scope "@nvidia/cuda" "echo \$PWD" | head -n1)/include

if [ -d "$NODE_CUDA_INCLUDE_DIR/thrust" ]; then rm "$NODE_CUDA_INCLUDE_DIR/thrust"; fi

ln -sf "$THRUST_INCLUDE_DIR" "$NODE_CUDA_INCLUDE_DIR/thrust"
