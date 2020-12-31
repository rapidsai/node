#!/usr/bin/env bash

set -Eeo pipefail

exec lerna run --scope '@nvidia/*' --stream "$@"
