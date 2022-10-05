#!/usr/bin/env bash

set -Eeo pipefail

rm -rf "$PWD/build"
mkdir -p "$PWD/build"

args="--stream --no-sort --parallel";

echo "running npm pack..."

lerna exec ${args} "npm pack --pack-destination $PWD/build \$PWD";

echo "running cpack..."

pkgs="$(lerna run ${args} --no-prefix --scope '@rapidsai/*' dev:cpack:enabled)";

args+=" $(for name in ${pkgs}; do echo "--scope $name"; done)";

lerna exec ${args} "\
cd build/Release \
&& cpack -G TGZ && rm -rf _CPack_Packages \
&& mv ./rapidsai_*-*-*.tar.gz \$LERNA_ROOT_PATH/build/"
