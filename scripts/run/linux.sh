#!/usr/bin/env -S bash -Eeo pipefail

exec lerna run $@ --stream --include-dependencies --scope '@nvidia/*' --scope '@rapidsai/*'
