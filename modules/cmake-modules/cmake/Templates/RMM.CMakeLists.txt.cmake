cmake_minimum_required(VERSION 3.12)

include(ExternalProject)

ExternalProject_Add(rmm
                    GIT_REPOSITORY    https://github.com/rapidsai/rmm.git
                    GIT_TAG           "${RMM_GIT_BRANCH_NAME}"
                    GIT_SHALLOW       TRUE
                    GIT_CONFIG        "advice.detachedhead=false"
                    SOURCE_DIR        "${RMM_ROOT}/rmm"
                    BINARY_DIR        "${RMM_ROOT}/build"
                    INSTALL_DIR       "${CMAKE_CURRENT_BINARY_DIR}"
                    CMAKE_ARGS        ${RMM_CMAKE_ARGS} -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR})
