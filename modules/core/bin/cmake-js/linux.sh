#!/usr/bin/env bash

set -Eeo pipefail

args=""
debug=false
compile_commands_json=

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -D|--debug) debug=true;;
        *) args="${args:+$args }$1";;
    esac; shift;
done

if [[ "$debug" == "false" ]]; then
    args="${args:+$args }-O build/Release";
    compile_commands_json="build/Release/compile_commands.json";
else
    args="${args:+$args }-D -O build/Debug";
    compile_commands_json="build/Debug/compile_commands.json";
fi

RAPIDS_CORE_HOME=$(dirname $(realpath "$0"))
RAPIDS_CORE_HOME=$(realpath "$RAPIDS_CORE_HOME/../../")

JOBS=$(node -e "console.log(require('os').cpus().length)") \
    PARALLEL_LEVEL=$JOBS CMAKE_BUILD_PARALLEL_LEVEL=$JOBS  \
    HOME="$RAPIDS_CORE_HOME"                               \
    npx cmake-js $args                                     \
 && ln -f -s $compile_commands_json compile_commands.json

if [[ "$debug" == "false" ]]; then
    if [[ $(basename $(realpath -m "$RAPIDS_CORE_HOME/../")) == "modules" ]]; then
        if [[ "$(which jq)" != "" ]]; then
            jq -s '.|flatten' \
                $(find .. -type f -path "*$compile_commands_json") \
            > "../../compile_commands.json"
        fi
    fi
fi
