cmake_minimum_required(VERSION 3.12)

include(ExternalProject)

ExternalProject_Add(GLEW
                    GIT_REPOSITORY    https://github.com/Perlmint/glew-cmake.git
                    GIT_TAG           "${GLEW_GIT_BRANCH_NAME}"
                    GIT_SHALLOW       TRUE
                    GIT_CONFIG        "advice.detachedhead=false"
                    SOURCE_DIR        "${GLEW_ROOT}/glew"
                    BINARY_DIR        "${GLEW_ROOT}/build"
                    INSTALL_DIR       "${CMAKE_CURRENT_BINARY_DIR}"
                    CMAKE_ARGS        ${GLEW_CMAKE_ARGS} -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR})
