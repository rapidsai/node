#!/usr/bin/env bash

set -Eeo pipefail

if ! type jq >/dev/null 2>&1; then
    exit 0;
fi

cwd="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )";

RAPIDS_MODULES_PATH="$(realpath "$cwd/../../../")";

if [[ "$(basename "$RAPIDS_MODULES_PATH")" != modules ]]; then
    exit 0;
fi

# Get the latest `compile_commands.json` from each module
compile_command_files="$(                                   \
find "$RAPIDS_MODULES_PATH" -mindepth 1 -maxdepth 1 -type d | xargs -n1 -I__ \
    bash -c 'find __ -type f \
        -path "*build/*/compile_commands.json" \
        -exec stat -c "%y %n" {} + \
  | sort -r \
  | head -n1' \
  | grep -Eo "$RAPIDS_MODULES_PATH/.*$" || echo "" \
)";

# Now merge them all together
jq -s '.|flatten' $(echo "$compile_command_files")    \
    > "$RAPIDS_MODULES_PATH/../compile_commands.json" \
 || true;
