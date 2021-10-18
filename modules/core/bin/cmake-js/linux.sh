#!/usr/bin/env bash

set -Eeo pipefail

if [[ "$(yarn why cmake-js >/dev/null 2>&1)" || $? != 0 ]]; then exit 0; fi;

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
    args="${args:+$args }--CDCMAKE_LIBRARY_OUTPUT_DIRECTORY=$(realpath -m build/Release)";
else
    args="${args:+$args }-D -O build/Debug";
    compile_commands_json="build/Debug/compile_commands.json";
    args="${args:+$args }--CDCMAKE_LIBRARY_OUTPUT_DIRECTORY=$(realpath -m build/Debug)";
fi

RAPIDS_CORE_PATH=$(dirname $(realpath "$0"))
RAPIDS_CORE_PATH=$(realpath "$RAPIDS_CORE_PATH/../../")
RAPIDS_MODULES_PATH=$(realpath "$RAPIDS_CORE_PATH/../")

echo "\
=====================================================
PARALLEL_LEVEL=${PARALLEL_LEVEL:-1}
====================================================="

PARALLEL_LEVEL=${PARALLEL_LEVEL:-1}                    \
CMAKE_BUILD_PARALLEL_LEVEL=${PARALLEL_LEVEL:-1}        \
HOME="$RAPIDS_CORE_PATH"                               \
    cmake-js ${args} | grep -v -P '^ptxas /tmp/tmpxft(.*?)$'

if [[ "$debug" == "false" ]]; then
    if [[ $(basename $RAPIDS_MODULES_PATH) == "modules" ]]; then
        if [[ "$(which jq)" != "" ]]; then
            jq -s '.|flatten' \
                $(find .. -type f -path "*$compile_commands_json") \
            > "../../compile_commands.json" || true
        fi
    fi
fi
