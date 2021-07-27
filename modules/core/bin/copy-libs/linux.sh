#!/usr/bin/env bash

set -Eeo pipefail

copy_libs() {
    (
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
            deps=$(ldd "$lib" \
                | grep --color=never -iE '*.so(.*?) => /opt/rapids/node' \
                | sed -r 's@^.*?(/opt/rapids/node/.*\.so*[^\s\(]*?).*?$@\1@' \
                | tr -d ' ' || echo "")
            for dep in ${deps}; do copy_libs "$dep" "$dir"; done;
        fi
    )
}

dir=""
libs=""

for lib in ${@}; do
    if [[ "${lib##*.}" == "node" ]]; then
        lib="$PWD/$lib"
        dir="$(dirname $(realpath -m "$lib"))"
        mkdir -p "$dir"
    elif [[ -e "/opt/rapids/node/modules/.cache/build" ]]; then
        lib="$(shopt -s globstar; cd /opt/rapids/node/modules/.cache && realpath -m $lib)"
    else
        lib="$(shopt -s globstar; realpath -m $lib)"
    fi
    libs="${libs:+$libs }$(echo $lib)"
done

echo "rm -rf $dir"/*.so* && rm -rf "$dir"/*.so*
ls -l "$dir/*".so* 2>/dev/null || true

for lib_ in ${libs}; do
    echo "copying lib $lib_";
    copy_libs "$lib_" "$dir";
done;
