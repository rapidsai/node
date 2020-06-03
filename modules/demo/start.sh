#!/usr/bin/env bash

set -Eeo pipefail

cd $(dirname "$(realpath "$0")")

stat ../../node_modules/esm/node_modules/.cache &> /dev/null \
 && rm -rf ../../node_modules/esm/node_modules/.cache

FILE=${1:-"basic"};

if [[ ! -z "${FILE// }" ]]; then shift; fi;

FILE=$(node -e "console.log(require.resolve('./$FILE'))")

DIR=$(dirname "$FILE")
FILE=$(basename "$FILE")

cd "$DIR"

exec node -r esm --trace-uncaught $FILE "$@"

# exec node -r esm --trace-uncaught --abort-on-uncaught-exception $FILE "$@"
