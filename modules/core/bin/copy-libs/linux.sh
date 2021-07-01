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

dir=""
libs=""

for lib in ${@}; do
    if [[ "${lib##*.}" == "node" ]]; then
        lib="$PWD/$lib"
        dir="$(dirname $(realpath -m "$lib"))"
        mkdir -p "$dir"
    elif [[ -e "/opt/node-rapids/modules/.cache" ]]; then
        lib="/opt/node-rapids/modules/.cache/$lib"
    else
        lib="$PWD/$lib"
    fi
    libs="${libs:+$libs }$(echo $lib)"
done

for lib in ${libs}; do
    echo "copying lib $lib";
    copy_libs "$lib" "$dir";
done;
