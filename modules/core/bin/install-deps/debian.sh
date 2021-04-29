#!/usr/bin/env bash

set -Eeo pipefail

APT_DEPS=""
CMAKE_OPTS=""
INSTALL_CMAKE=""
INSTALLED_CLANGD=""
OS_RELEASE=$(lsb_release -cs)
IS_UBUNTU=$(. /etc/os-release;[ "$ID" = "ubuntu" ] && echo 1 || echo 0)

ask_before_install() {
    while true; do
        read -p "$1 " CHOICE </dev/tty
        case $CHOICE in
            [Nn]* ) break;;
            [Yy]* ) eval $2; break;;
            * ) echo "Please answer 'y' or 'n'";;
        esac
    done
}

check_apt() {
    [ -n "$(apt policy $1 2> /dev/null | grep -i 'Installed: (none)')" ] && echo "0" || echo "1";
}

install_apt_deps() {
    for DEP in $@; do
        if [ "$(check_apt $DEP)" -eq "0" ]; then
            ask_before_install \
            "Missing $DEP. Install $DEP now? (y/n)" \
                "APT_DEPS=\${APT_DEPS:+\$APT_DEPS }$DEP";
        fi;
    done
}

install_cmake() {
    INSTALL_CMAKE=1
    install_apt_deps zlib1g-dev
}

install_vscode() {
    APT_DEPS="${APT_DEPS:+$APT_DEPS }code"
    if [ ! -d "/etc/apt/sources.list.d/vscode.list" ]; then
        curl https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
        sudo install -o root -g root -m 644 packages.microsoft.gpg /usr/share/keyrings/ && rm packages.microsoft.gpg
        sudo sh -c 'echo "deb [arch=amd64 signed-by=/usr/share/keyrings/packages.microsoft.gpg] https://packages.microsoft.com/repos/vscode stable main" > /etc/apt/sources.list.d/vscode.list'
    fi
}

install_clangd() {
    INSTALLED_CLANGD=1
    APT_DEPS="${APT_DEPS:+$APT_DEPS }clangd-12 clang-format-12"
    if [ ! -d "/etc/apt/sources.list.d/llvm.list" ]; then
        curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
        echo "deb http://apt.llvm.org/$OS_RELEASE/ llvm-toolchain-$OS_RELEASE-12 main
deb-src http://apt.llvm.org/$OS_RELEASE/ llvm-toolchain-$OS_RELEASE-12 main
" | sudo tee /etc/apt/sources.list.d/llvm.list
    fi
}

install_vscode_extensions() {
    CODE="$1"
    for EXT in ${@:2}; do
        if [ -z "$($CODE --list-extensions | grep $EXT)" ]; then
            ask_before_install \
                "Missing $CODE extension $EXT. Install $EXT now? (y/n)" \
                "$CODE --install-extension $EXT"
        fi
    done
}

[ -z "$(which cmake)" ] && ask_before_install "Missing cmake. Install cmake (y/n)?" "install_cmake"
[ -z "$(which code)" ]  && ask_before_install "Missing vscode. Install vscode (y/n)?" "install_vscode"
[ -z "$(which clangd)" ] && ask_before_install "Missing clangd. Install clangd (y/n)?" "install_clangd"

install_apt_deps jq software-properties-common \
    # cuSpatial dependencies
    libgdal-dev \
    # cuDF dependencies
    libboost-filesystem-dev \
    # X11 dependencies
    libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev \
    # node-canvas dependencies
    libcairo2-dev libpango1.0-dev libjpeg-dev libgif-dev librsvg2-dev \
    # GLEW dependencies
    build-essential libxmu-dev libxi-dev libgl1-mesa-dev libegl1-mesa-dev libglu1-mesa-dev

if [ -n "$INSTALL_CMAKE" ]; then
    # Ensure Qt-5 is installed for CMake GUI, otherwise no cmake GUI
    if [ "$(check_apt qt5-default)" -eq "0" ]; then
        ask_before_install "Missing qt5-default needed by CMake GUI. Install it now (y/n)?" \
            'APT_DEPS=${APT_DEPS:+$APT_DEPS }qt5-default; CMAKE_OPTS=${CMAKE_OPTS:+$CMAKE_OPTS }--qt-gui;';
    fi

    if [ "$(check_apt libcurl4-openssl-dev)" -eq "0" ] || [ "$(check_apt libssl-dev)" -eq "0" ]; then
        ask_before_install "Missing libcurl4-openssl-dev and/or libssl-dev needed by CMake. Install them now (y/n)?" \
            'APT_DEPS=${APT_DEPS:+$APT_DEPS }libcurl4-openssl-dev libssl-dev; CMAKE_OPTS=${CMAKE_OPTS:+$CMAKE_OPTS }--system-curl;';
    fi
fi

if [ -n "$APT_DEPS" ]; then
    sudo apt update
    sudo apt install -y $APT_DEPS;
    if [ -n "$INSTALLED_CLANGD" ]; then
        sudo update-alternatives --install /usr/bin/clangd clangd /usr/bin/clangd-12 100
        sudo update-alternatives --set clangd /usr/bin/clangd
    fi
fi

if [ -n "$INSTALL_CMAKE" ]; then
    NJOBS=$(nproc --ignore=2)
    CMAKE_OPTS="${CMAKE_OPTS:+$CMAKE_OPTS } --parallel=$NJOBS"
    CMAKE_VERSION=$(curl -s https://api.github.com/repos/Kitware/CMake/releases/latest | jq -r ".tag_name" | tr -d 'v')
    wget -O- https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}.tar.gz | tar -xz \
     && cd cmake-${CMAKE_VERSION} \
        && ./bootstrap $CMAKE_OPTS \
        && sudo make install -j${NJOBS} \
     && cd - && rm -rf cmake-${CMAKE_VERSION}
fi

for CODE in "code" "code-insiders"; do
    # 1. Install Microsoft C++ Tools if it isn't installed
    # 2. Install vscode-clangd if it isn't installed
    if [ -n "$(which $CODE)" ]; then
        install_vscode_extensions "$CODE" "ms-vscode.cpptools" "llvm-vs-code-extensions.vscode-clangd";
    fi
done
