#!/usr/bin/env bash

set -Eeuo pipefail

cd "$(dirname "$(realpath "$0")")" || exit 1

mkdir -p ../../../.aws
touch ../../../.aws/credentials

docker build -f Dockerfile -t node-rapids-generate-creds .

declare -a args=()

if test -d "$HOME/.config/gh"; then
    args+=(-v "$HOME/.config/gh:/gh:ro")
fi

if test -d "$HOME/.ssh"; then
    args+=(-v "$HOME/.ssh:/ssh:ro")
fi

if test -f "$HOME/.ssh/known_hosts"; then
    args+=(-v "$HOME/.ssh/known_hosts:/ssh/known_hosts:rw")
fi

if test -v SSH_AUTH_SOCK && test -e "$SSH_AUTH_SOCK"; then
    args+=(
        -v "$SSH_AUTH_SOCK:/tmp/ssh_auth_sock" \
        -e "SSH_AUTH_SOCK=/tmp/ssh_auth_sock" \
    )
fi

args+=(
  -v "$(realpath -m ../../../.aws/credentials):/out/credentials:rw" \
  -v "$(realpath -m ../../../.config/sccache/config):/out/sccache:rw" \
#   -v "$(pwd)/generate.sh:/generate.sh:rw" \
  -e "PUID=$(id -u)" \
  -e "PGID=$(id -g)" \
  -e "RUN_AS_USER=$(whoami)" \
  -e "GH_TELEMETRY=false" \
  -e "GH_TOKEN=$(gh auth token)" \
)

# docker run --rm -it "${args[@]}" node-rapids-generate-creds /bin/bash
docker run --rm -it "${args[@]}" node-rapids-generate-creds /generate.sh
