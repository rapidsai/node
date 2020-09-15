#!/usr/bin/env bash

set -Eeo pipefail

echo "cleaning rapids-js"

if [[ ! -d node_modules || ! -d node_modules/lerna || ! -d node_modules/rimraf ]]; then
    yarn --silent --non-interactive --no-node-version-check --ignore-engines;
fi

# clean modules/*/build dirs
npx lerna run --no-bail clean || true;
npx lerna clean --loglevel error --yes || true;
rm -rf yarn.lock build node_modules compile_commands.json
