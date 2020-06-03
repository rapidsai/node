cmake_minimum_required(VERSION 3.12)

include(ExternalProject)

ExternalProject_Add(GLFW
                    GIT_REPOSITORY    https://github.com/glfw/glfw.git
                    GIT_TAG           "${GLFW_GIT_BRANCH_NAME}"
                    GIT_SHALLOW       TRUE
                    GIT_CONFIG        "advice.detachedhead=false"
                    SOURCE_DIR        "${GLFW_ROOT}/glfw"
                    BINARY_DIR        "${GLFW_ROOT}/build"
                    INSTALL_DIR       "${CMAKE_CURRENT_BINARY_DIR}"
                    CMAKE_ARGS        ${GLFW_CMAKE_ARGS} -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR})
