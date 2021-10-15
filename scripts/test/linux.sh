#!/usr/bin/env -S bash -Eeo pipefail

exec lerna run --no-bail --scope '@nvidia/*' --scope '@rapidsai/*' --stream --concurrency 1 test
