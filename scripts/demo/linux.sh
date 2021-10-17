#!/usr/bin/env bash

set -Eeo pipefail

find node_modules -name .cache -type d -exec rm -rf "{}" +

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
    $(echo modules/demo/{graph,luma,spatial,xterm,client-server,umap,viz-app}/package.json)
    $(find modules/demo/{deck,tfjs,ipc,ssr,sql} -maxdepth 2 -type f -name 'package.json')
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
    echo "\`yarn demo $DEMO${@:+ ${@:-}}\`"
fi

ARGS="${@:-}";

if [[ "$DEMO" =~ "modules/demo/luma" ]]; then ARGS="${@:-01}";
elif [[ "$DEMO" =~ "modules/demo/ipc/umap" ]]; then ARGS="${@:-tcp://0.0.0.0:6000}";
fi

if [[ "$DEMO" =~ "modules/demo/client-server" ]]; then
    NODE_ENV=${NODE_ENV:-production} \
    NODE_NO_WARNINGS=${NODE_NO_WARNINGS:-1} \
    exec npm --prefix="$DEMO" ${ARGS} start
elif [[ "$DEMO" =~ "modules/demo/deck/playground-ssr" ]]; then
    NODE_ENV=${NODE_ENV:-production} \
    NODE_NO_WARNINGS=${NODE_NO_WARNINGS:-1} \
    exec npm --prefix="$DEMO" ${ARGS} start
else
    NODE_ENV=${NODE_ENV:-production} \
    NODE_NO_WARNINGS=${NODE_NO_WARNINGS:-1} \
    exec npm --prefix="$DEMO" start -- ${ARGS}
    # exec node --experimental-vm-modules --trace-uncaught "$DEMO" ${ARGS}
fi
