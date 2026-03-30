#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
IS_DEBIAN="$(. /etc/os-release; [ "$ID_LIKE" = "debian" ] && echo true || echo false)"

if $IS_DEBIAN; then
    source "$SCRIPT_DIR/debian.sh"
fi

# TODO: other distros
