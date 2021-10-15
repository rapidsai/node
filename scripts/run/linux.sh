#!/usr/bin/env bash

set -Eeo pipefail

exec lerna run $@ --stream --include-dependencies --scope '@rapidsai/*'
