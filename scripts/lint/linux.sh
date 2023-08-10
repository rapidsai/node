#!/usr/bin/env bash

set -Eeo pipefail

args="";
fix_arg="";
JOBS="${JOBS:-}";

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -j*)
            J="${1#-j}";
            if [[ ${J} =~ ^[[:digit:]]+$ ]]; then
                JOBS="${J}";
            fi;;
        --fix) fix_arg="$1";;
        *) args="${args:+$args }$1";;
    esac; shift;
done

JOBS=${JOBS:-${PARALLEL_LEVEL:-$(nproc --ignore=2)}};

tsc_files="";
cpp_files="";
cmd_input="$(echo "$args" | tr ' ' '\n')";
tsc_regex="^(\./)?modules/\w+?/(src|test)/.*?\.ts$";
cpp_regex="^(\./)?modules/\w+?/(src|include)/.*?\.(h|cc?|cuh?|(c|h)pp)$";

if [[ "$cmd_input" != "" ]]; then
    tsc_files=$(echo "$cmd_input" | grep -xiE --color=never "$tsc_regex" || echo "");
    cpp_files=$(echo "$cmd_input" | grep -xiE --color=never "$cpp_regex" || echo "");
else
    tsc_files=$(find . -type f -regextype posix-extended -iregex "$tsc_regex" || echo "");
    cpp_files=$(find . -type f -regextype posix-extended -iregex "$cpp_regex" || echo "");
fi

echo "Running clang-format...";
time clang-format-17 -i $cpp_files $tsc_files;
echo "";

echo "Running ESLint (on up to $JOBS cores)...";
time echo $tsc_files | xargs -P$JOBS -n1 \
    node_modules/.bin/eslint --ignore-path .gitignore ${fix_arg};
echo "";
