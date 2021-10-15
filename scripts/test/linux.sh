#!/usr/bin/env bash

set -Eeo pipefail

exec lerna run --no-bail --scope '@rapidsai/*' --stream --concurrency 1 test
