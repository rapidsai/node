#!/usr/bin/env bash

set -Eeo pipefail

TOP="$(pwd)"
BIN="$(realpath node_modules/.bin)"
DIRS=$(lerna exec --scope "@rapidsai/*" "echo \$PWD")
RAPIDS_CORE_PATH=$(lerna exec --scope "@rapidsai/core" "echo \$PWD" | head -n1)

# ensure the cache dirs exist (clangd index, etc.)
mkdir -p "$TOP"/.cache/{binary,clangd,source}

for DIR in $DIRS; do
    # symlink node_modules/.bin dirs to the root node_modules/.bin
    mkdir -p "$DIR/node_modules"
    if [[ "$BIN" != $DIR/node_modules/.bin ]]; then
        rm -rf "$DIR/node_modules/.bin"
        ln -sf "$BIN" "$DIR/node_modules/.bin"
        # copy the ESLint settings file (for the VSCode ESLint plugin)
        cp ".eslintrc.js" "$DIR/.eslintrc.js"
        # remove the local .cache symlink
        rm -rf "$DIR/.cache"
        # symlink to the shared top-level .cache dir
        ln -sf "$(realpath --relative-to="$DIR" "$TOP/.cache")" "$DIR/.cache"
        # symlink to the shared .env settings file
        touch ".env" && ln -sf "$(realpath --relative-to="$DIR" "$TOP/.env")" "$DIR/.env"
        # symlink to the shared .clangd settings file
        touch ".clangd" && ln -sf "$(realpath --relative-to="$DIR" "$TOP/.clangd")" "$DIR/.clangd"
    fi;
done

# use `which npm` because yarn prepends its own path to /tmp/yarn-XXX/node
NPM_BIN_PATH="${npm_node_execpath:-$(which npm)}"
NAPI_INCLUDE_DIR="$PWD/node_modules/node-addon-api"
NODE_INCLUDE_DIR="${NPM_BIN_PATH%/bin/npm}/include"

# symlink node headers
ln -sf "$NODE_INCLUDE_DIR/node/node_api.h" "$RAPIDS_CORE_PATH/include/node_api.h"
# symlink napi headers
ln -sf "$NAPI_INCLUDE_DIR/napi.h" "$RAPIDS_CORE_PATH/include/napi.h"
ln -sf "$NAPI_INCLUDE_DIR/napi-inl.h" "$RAPIDS_CORE_PATH/include/napi-inl.h"
ln -sf "$NAPI_INCLUDE_DIR/napi-inl.deprecated.h" "$RAPIDS_CORE_PATH/include/napi-inl.deprecated.h"
