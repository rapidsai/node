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
        --)
            break
            ;;
        *)
            cmd_args+=("$1")
            shift
            ;;
    esac
done

if test ${#scope_args[@]} -eq 0; then
    scope_args+=(--scope '@rapidsai/*')
fi

exec lerna run --concurrency 1 --no-bail --stream "${scope_args[@]}" "${cmd_args[@]}" test "$@"
