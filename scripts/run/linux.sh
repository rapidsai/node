#!/usr/bin/env bash

set -Eeuo pipefail

cmd_args=()
scope_args=()

while test -n "${1:+x}"; do
    case "$1" in
        --scope)
            scope_args+=("$1" "$2")
            shift 2
            ;;
        --parallel)
            export CMAKE_BUILD_PARALLEL_LEVEL="$2"
            cmd_args+=("$1" "$2")
            shift 2
            ;;
        *)
            cmd_args+=("$1")
            shift
            ;;
    esac
done

if test ${#scope_args[@]} -eq 0; then
    scope_args+=(--scope '@rapidsai/*' --include-dependencies)
fi

exec lerna run --stream "${scope_args[@]}" "${cmd_args[@]}" "$@"
