#!/usr/bin/env bash

set -Eeo pipefail

exec lerna run --no-bail --scope '@nvidia/*' --stream --concurrency 1 test
