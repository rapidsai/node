#!/usr/bin/env bash

set -Eeo pipefail

rm -rf build
mkdir -p build

# exec npm pack --workspaces --pack-destination "$PWD/build"

exec lerna exec \
    --no-sort --stream --loglevel error \
    "npm pack --pack-destination $PWD/build \$PWD"

# exec lerna exec \
#     --no-sort --stream --loglevel error \
#     --scope '@nvidia/*' --scope '@rapidsai/*' \
#     "npm pack --pack-destination $PWD/build \$PWD"
