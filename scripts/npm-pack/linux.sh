#!/usr/bin/env bash

set -Eeuo pipefail

rm -rf "$PWD/build"
mkdir -p "$PWD/build"

declare -a args=(--stream --no-sort --parallel);

echo "running npm pack..."

lerna exec "${args[@]}" "npm pack --pack-destination $PWD/build \$PWD";

echo "running cpack..."

declare -a pkgs="($(lerna run "${args[@]}" --no-prefix --scope '@rapidsai/*' dev:cpack:enabled))";

mapfile -O "${#args[@]}" -t args < <(echo "${pkgs[*]/#/--scope }" | tr ' ' '\n')

lerna exec "${args[@]}" "\
cd _build/Release \
&& cpack -G TGZ && rm -rf _CPack_Packages \
&& mv ./rapidsai_*-*-*.tar.gz \$LERNA_ROOT_PATH/build/"
