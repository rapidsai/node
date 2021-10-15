#!/usr/bin/env bash

set -Eeo pipefail

rm -rf "$PWD/build"
mkdir -p "$PWD/build"
exec lerna exec --no-sort --stream "npm pack --pack-destination $PWD/build \$PWD"
