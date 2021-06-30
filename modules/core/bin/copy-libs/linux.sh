#!/usr/bin/env bash

set -Eeo pipefail

copy_libs() {
    lib="$1"
    dir="$2"
    if [[ -e "$lib" ]]; then
        src="$lib"
        dst="$dir/$(basename $lib)"
        if [[ "$src" != "$dst" ]]; then
            echo "cp -d $src $dst" && cp -d "$src" "$dst"
            while [[ -L "$lib" ]]; do
                dst="$(readlink $lib)"
                lib="$(cd $(dirname "$lib") && realpath -s "$dst")"
                echo "cp -d $lib $dir/$(basename $lib)" \
                    && cp -d "$lib" "$dir/$(basename $lib)"
            done
        fi
        libs=$(ldd "$lib" \
            | grep --color=never -iE '*.so(.*?) => /opt/node-rapids' \
            | sed -r 's@^.*?(/opt/node-rapids/.*\.so*[^\s\(]*?).*?$@\1@' \
            | tr -d ' ' || echo "")
        for lib in ${libs}; do copy_libs "$lib" "$dir"; done;
    fi
}

for lib in ${@}; do
    if [[ -e "$PWD/$lib" ]]; then
        lib="$PWD/$lib";
    elif [[ -e "/opt/node-rapids/modules/.cache/$lib" ]]; then
        lib="/opt/node-rapids/modules/.cache/$lib";
    fi
    if [[ "${lib##*.}" == "node" ]]; then
        dir="$(dirname $(realpath -m "$lib"))"
        mkdir -p "$dir"
    fi
    echo "copying lib $lib";
    copy_libs "$lib" "$dir";
done;
