#!/usr/bin/env bash

set -Eeo pipefail

_copy_lib() {
    (
        src="$1"
        dst="$2"
        if [ ! -e "$dst" ]; then
            echo "${3:+$3 }cp -d $src $dst";
            cp -d "$src" "$dst";
        fi
    )
}

_copy_lib_links() {
    (
        lib="$1"
        dir="$2"
        if [[ -e "$lib" ]]; then
            src="$lib"
            dst="$dir/$(basename $lib)"
            if [[ "$src" != "$dst" ]]; then
                _copy_lib "$src" "$dst";
                while [[ -L "$lib" ]]; do
                    dst="$(readlink $lib)"
                    lib="$(cd $(dirname "$lib") && realpath -s "$dst")"
                    _copy_lib "$lib" "$dir/$(basename $lib)";
                done
            fi
        fi
    )
}

_collect_dependencies() {
    (
        libs=""
        lib="$1"
        dir="$2"
        if [[ -e "$lib" ]]; then
            libs="$lib"
            deps=$(ldd "$lib" \
                | grep --color=never -iE '*.so(.*?) => /opt/rapids/node' \
                | sed -r 's@^.*?(/opt/rapids/node/.*\.so*[^\s\(]*?).*?$@\1@' \
                | tr -d ' ' || echo "")
            for dep in ${deps}; do
                libs="$libs\n$(_collect_dependencies "$dep" "$dir")";
            done;
        fi
        echo -e "$libs" | sort | uniq;
    )
}

dir_=""
libs=""

for lib in ${@}; do
    if [[ "${lib##*.}" == "node" ]]; then
        lib="$PWD/$lib"
        dir_="$(dirname $(realpath -m "$lib"))"
        mkdir -p "$dir_"
    elif [[ -e "/opt/rapids/node/modules/.cache/build" ]]; then
        lib="$(shopt -s globstar; cd /opt/rapids/node/modules/.cache && realpath -m $lib)"
    else
        lib="$(shopt -s globstar; realpath -m $lib)"
    fi
    libs="${libs:+$libs }$(echo $lib)"
done

echo "rm -rf $dir_"/*.so* && rm -rf "$dir_"/*.so*

ls -l "$dir_/*".so* 2>/dev/null || true

deps=""

for lib_ in ${libs}; do
    echo "copying dependencies of: $lib_";
    deps="${deps:+$deps\n}$(_collect_dependencies "$lib_" "$dir_")";
    deps="$(echo -e "$deps" | sort | uniq)";
done

# echo -e "deps:\n$deps"

for dep_ in ${deps}; do
    _copy_lib_links "$dep_" "$dir_";
done
