#!/usr/bin/env bash

set -Eeo pipefail

exec lerna run --no-bail --scope '@nvidia/*' --scope '@rapidsai/*' --stream --concurrency 1 test
