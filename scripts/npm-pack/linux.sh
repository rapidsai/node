#!/usr/bin/env bash

set -Eeo pipefail

rm -rf "$PWD/build" && mkdir -p "$PWD/build" \
 && lerna_args="--no-sort --stream --parallel --no-prefix" \
 && echo "running cpack" \
 && cp $(lerna run ${lerna_args} --scope '@rapidsai/*' dev:cpack) "$PWD/build" \
 && echo "running npm pack" \
 && lerna exec ${lerna_args} "npm pack --pack-destination $PWD/build \$PWD";
