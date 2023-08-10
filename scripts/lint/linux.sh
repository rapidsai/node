#!/usr/bin/env bash

set -Eeo pipefail

args="";
fix_="";
jobs="${JOBS:-${PARALLEL_LEVEL:-$(nproc --ignore=2)}}";

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -j*)
            J="${1#-j}";
            if [[ ${J} =~ ^[[:digit:]]+$ ]]; then
                jobs="${J}";
            else
                jobs="$(nproc --ignore=2)";
            fi;;
        --fix) fix_="$1";;
        *) args="${args:+$args }$1";;
    esac; shift;
done

tsc_files="";
cpp_files="";
cmd_input="$(tr ' ' '\n' <<< "$args")";
tsc_regex="^(\.\/)?modules\/\w+?\/(src|test)\/.*?\.ts$";
cpp_regex="^(\.\/)?modules\/\w+?\/(src|include)\/.*?\.(h|cc?|cuh?|(c|h)pp)$";

if test -n "$(head -n1 <<< "$cmd_input")"; then
    tsc_files="$(grep -Eiox --color=never "$tsc_regex" <<<  "$cmd_input" || echo "")";
    cpp_files="$(grep -Eiox --color=never "$cpp_regex" <<<  "$cmd_input" || echo "")";
else
    tsc_files="$(find . -type f -regextype posix-extended -iregex "$tsc_regex" || echo "")";
    cpp_files="$(find . -type f -regextype posix-extended -iregex "$cpp_regex" || echo "")";
fi

echo "Running clang-format...";
time                                              \
    xargs -d'\n' -t -n1 -I% -P$jobs               \
        <<< "$(echo -e "$cpp_files\n$tsc_files")" \
    clang-format-17 -i %;
echo "";

echo "Running ESLint (on up to $jobs cores)...";
time                                                         \
    xargs -d'\n' -n1 -I% -P$jobs                             \
        <<< "$tsc_files"                                     \
    node_modules/.bin/eslint --ignore-path .gitignore $fix_ %;
echo "";
