#!/usr/bin/env bash

set -Eeo pipefail

npx lerna run --scope '@nvidia/*' --stream --concurrency 1 test
