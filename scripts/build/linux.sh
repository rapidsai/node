#!/usr/bin/env bash

set -Eeo pipefail

CMD="build"

if [[ "$1" == "--fast" ]];
then CMD="compile";
elif [[ "$1" == "--clean" ]];
then CMD="rebuild";
else CMD="build";
fi

npx lerna run --scope '@nvidia/*' --stream $CMD
