#!/usr/bin/env bash

set -Eeo pipefail

npx lerna run --no-bail --scope '@nvidia/*' --stream --concurrency 1 test
