#!/usr/bin/env bash

set -Eeo pipefail

JOBS=$(node -e "console.log(require('os').cpus().length)") \
    PARALLEL_LEVEL=$JOBS CMAKE_BUILD_PARALLEL_LEVEL=$JOBS  \
    npx cmake-js $@                                        \
 && ln -f -s build/compile_commands.json compile_commands.json
