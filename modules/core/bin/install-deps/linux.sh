#!/usr/bin/env bash

set -Eeo pipefail


SCRIPT_DIR=$(dirname "$(realpath "$0")")
IS_DEBIAN=$(. /etc/os-release;[ "$ID_LIKE" = "debian" ] && echo 1 || echo 0)

if [ $IS_DEBIAN ]; then
    source "$SCRIPT_DIR/debian.sh"
fi

# TODO: other distros
