#!/usr/bin/env -S bash -Eeo pipefail

tsc_files=""
cpp_files=""
cmd_input="$(echo "$@" | tr ' ' '\n')"
tsc_regex="^(\./)?modules/\w+?/(src|test)/.*?\.ts$"
cpp_regex="^(\./)?modules/\w+?/(src|include)/.*?\.(h|cc?|cuh?|(c|h)pp)$"

if [[ "$cmd_input" != "" ]]; then
    tsc_files=$(echo "$cmd_input" | grep -xiE --color=never "$tsc_regex" || echo "")
    cpp_files=$(echo "$cmd_input" | grep -xiE --color=never "$cpp_regex" || echo "")
else
    tsc_files=$(find . -type f -regextype posix-extended -iregex "$tsc_regex" || echo "")
    cpp_files=$(find . -type f -regextype posix-extended -iregex "$cpp_regex" || echo "")
fi

J=$(nproc --ignore=2)

echo "Running clang-format..."
time clang-format-12 -i $cpp_files $tsc_files
echo ""

echo "Running ESLint (on up to $J cores)..."
time echo $tsc_files | xargs -P$J -n1 \
    node_modules/.bin/eslint --ignore-path .gitignore --fix
echo ""
