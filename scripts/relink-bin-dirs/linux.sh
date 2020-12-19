#!/usr/bin/env bash

set -Eeo pipefail

DIR="$(pwd)"
BIN="$(realpath node_modules/.bin)"
DIRS=$(npx lerna exec --scope "@nvidia/*" "echo \$PWD")
RAPIDS_CORE_PATH=$(npx lerna exec --scope "@nvidia/rapids-core" "echo \$PWD" | head -n1)
RAPIDS_MODULES_PATH=$(realpath "$RAPIDS_CORE_PATH/../")

# ensure the rapids-core cache dirs exist (clangd index, etc.)
mkdir -p "$RAPIDS_MODULES_PATH/.cache/cpm" \
         "$RAPIDS_MODULES_PATH/.cache/ccache" \
         "$RAPIDS_MODULES_PATH/.cache/clangd" ;

for DIR in $DIRS; do
    # symlink node_modules/.bin dirs to the root node_modules/.bin
    mkdir -p "$DIR/node_modules"
    rm -rf "$DIR/node_modules/.bin"
    ln -sf "$BIN" "$DIR/node_modules/.bin"
    cp ".prettierrc" ".eslintrc.js" "$DIR/"
    # # remove and recreate the local .cache dir
    rm -rf "$DIR/.cache"
    # symlink to the shared .cache dir under modules
    ln -sf "$RAPIDS_MODULES_PATH/.cache" "$DIR/.cache"
done

NODE_BIN_PATH="$(which node)"
NAPI_INCLUDE_DIR="$PWD/node_modules/node-addon-api"
NODE_INCLUDE_DIR="${NODE_BIN_PATH%/bin/node}/include"

# symlink node headers
ln -sf "$NODE_INCLUDE_DIR/node/node_api.h" "$RAPIDS_CORE_PATH/include/node_api.h"
# symlink napi headers
ln -sf "$NAPI_INCLUDE_DIR/napi.h" "$RAPIDS_CORE_PATH/include/napi.h"
ln -sf "$NAPI_INCLUDE_DIR/napi-inl.h" "$RAPIDS_CORE_PATH/include/napi-inl.h"
ln -sf "$NAPI_INCLUDE_DIR/napi-inl.deprecated.h" "$RAPIDS_CORE_PATH/include/napi-inl.deprecated.h"
