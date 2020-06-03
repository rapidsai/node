#!/usr/bin/env bash

set -Eeo pipefail
# set -x

fuzzy-find() {
    (
        for p in ${@}; do
            path="${p#./}"; # remove leading ./ (if exists)
            ext="${p##*.}"; # extract extension (if exists)
            if [[ $ext == $p ]];
                then echo $(find .                -print0 | grep -FzZ $path | tr '\0' '\n');
                else echo $(find . -name "*.$ext" -print0 | grep -FzZ $path | tr '\0' '\n');
            fi;
        done
    )
}

DEMO=""
if [[ "$1" =~ "modules/demo" ]]; then
    DEMO="$(fuzzy-find "$1/package.json" || echo '')";
    DEMO="${DEMO%\/package.json}"
    shift;
fi

if [[ "$DEMO" == "" ]]; then
    DEMOS="
    modules/demo/luma/package.json
    modules/demo/umap/package.json
    modules/demo/xterm/package.json
    $(find modules/demo/deck -maxdepth 2 -type f -name 'package.json')
    ";
    DEMOS="$(echo -e "$DEMOS" | grep -v node_modules | sort -Vr)";
    DEMOS=(${DEMOS});
    DEMOS=("${DEMOS[@]/%\/package.json}")
    echo "Please select a demo to run out:"
    select DEMO in "${DEMOS[@]}" "Quit"; do
        if [[ $REPLY -lt $(( ${#DEMOS[@]}+1 )) ]]; then
            break;
        elif [[ $REPLY -eq $(( ${#DEMOS[@]}+1 )) ]]; then
            exit 0;
        else
            echo "Invalid option, please select a demo (or quit)"
        fi
    done;
    echo "Run this demo directly via:"
    echo "\`npm run demo $DEMO${@:+ ${@:-}}\`"
fi

ARGS="${@:-}";

if [[ "$DEMO" =~ "modules/demo/luma/index.js" ]]; then ARGS="${@:-01}";
elif [[ "$DEMO" =~ "modules/demo/umap/index.js" ]]; then ARGS="${@:-tcp://0.0.0.0:6000}";
fi

exec node -r esm --trace-uncaught "$DEMO" $ARGS
