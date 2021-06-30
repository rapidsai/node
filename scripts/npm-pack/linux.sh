#!/usr/bin/env bash

set -Eeo pipefail

rm -rf bin
mkdir -p bin

exec lerna exec \
    --no-sort --stream --loglevel error \
    --scope '@nvidia/*' --scope '@rapidsai/*' \
    "npm pack --pack-destination $PWD/bin \$PWD"
