#!/usr/bin/env bash

set -Eeo pipefail

# clean modules/*/build dirs
echo "cleaning rapids-js" \
 && npm install --no-save lerna@3.20.2 rimraf@3.0.0 \
 && npx lerna run --no-bail clean || true \
 && npx lerna clean --loglevel error --yes || true \
 && rm -rf yarn.lock build node_modules compile_commands.json
