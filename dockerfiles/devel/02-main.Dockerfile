ARG FROM_IMAGE

FROM ${FROM_IMAGE}

ENV NVIDIA_DRIVER_CAPABILITIES all

RUN export DEBIAN_FRONTEND=noninteractive \
 && apt update --fix-missing \
 && apt install -y --no-install-recommends \
    # X11 dependencies
    libxi-dev libxrandr-dev libxinerama-dev libxcursor-dev \
    # node-canvas dependencies
    libcairo2-dev libpango1.0-dev libjpeg-dev libgif-dev librsvg2-dev \
    # GLFW Wayland dependencies
    extra-cmake-modules libwayland-dev wayland-protocols libxkbcommon-dev \
    # GLEW dependencies
    build-essential libxmu-dev libxi-dev libgl1-mesa-dev libegl1-mesa-dev libglu1-mesa-dev \
    # cuSpatial dependencies
    libgdal-dev \
    # blazingSQL dependencies
    maven openjdk-8-jdk libboost-regex-dev libboost-system-dev libboost-filesystem-dev \
    # UCX build dependencies
    automake autoconf libtool \
    # UCX runtime dependencies
    libibverbs-dev librdmacm-dev libnuma-dev libhwloc-dev \
 # Install UCX
 && git clone --depth 1 --branch v1.10.x https://github.com/openucx/ucx.git /tmp/ucx \
 && curl -o /tmp/cuda-alloc-rcache.patch \
         -L https://raw.githubusercontent.com/rapidsai/ucx-split-feedstock/11ad7a3c1f25514df8064930f69c310be4fd55dc/recipe/cuda-alloc-rcache.patch \
 && cd /tmp/ucx && git apply /tmp/cuda-alloc-rcache.patch && rm /tmp/cuda-alloc-rcache.patch \
 && sed -i 's/io_demo_LDADD =/io_demo_LDADD = $(CUDA_LDFLAGS)/' /tmp/ucx/test/apps/iodemo/Makefile.am \
 && /tmp/ucx/autogen.sh && mkdir /tmp/ucx/build && cd /tmp/ucx/build \
 && ../contrib/configure-release \
    --prefix=/usr/local \
    --without-java --with-cuda=/usr/local/cuda \
    --enable-mt CPPFLAGS=-I/usr/local/cuda/include \
 && make -C /tmp/ucx/build -j install \
 # Clean up
 && apt autoremove -y && apt clean \
 && rm -rf \
    /tmp/* \
    /var/tmp/* \
    /var/lib/apt/lists/*

WORKDIR /opt/rapids/node

USER node
