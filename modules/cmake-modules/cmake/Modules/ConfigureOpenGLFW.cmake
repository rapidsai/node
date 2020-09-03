#=============================================================================
# Copyright (c) 2020, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

include(FindGLFW)

if(NOT GLFW_FOUND OR (GLFW_VERSION VERSION_LESS ${REQUIRED_GLFW_VERSION}))

    message(STATUS "GLFW ${REQUIRED_GLFW_VERSION} not found, building from source")

    set(GLFW_ROOT "${CMAKE_BINARY_DIR}/glfw")

    if(REQUIRED_GLFW_VERSION VERSION_EQUAL 3.3)
        set(GLFW_GIT_BRANCH_NAME "3.3-stable")
    else()
        set(GLFW_GIT_BRANCH_NAME "${REQUIRED_GLFW_VERSION}")
    endif()

    set(GLFW_CMAKE_ARGS " -DGLFW_INSTALL=ON"
                        " -DGLFW_BUILD_DOCS=OFF"
                        " -DGLFW_BUILD_TESTS=OFF"
                        " -DGLFW_BUILD_EXAMPLES=OFF"
                        " -DBUILD_SHARED_LIBS=${GLFW_USE_SHARED_LIBS}")

    configure_file("${CMAKE_CURRENT_LIST_DIR}/../Templates/GLFW.CMakeLists.txt.cmake"
                   "${GLFW_ROOT}/CMakeLists.txt")

    execute_process(
        COMMAND ${CMAKE_COMMAND} -Wno-dev -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE GLFW_CONFIG
        WORKING_DIRECTORY ${GLFW_ROOT})

    if(GLFW_CONFIG)
        message(FATAL_ERROR "Configuring GLFW failed: " ${GLFW_CONFIG})
    endif(GLFW_CONFIG)

    execute_process(
        COMMAND ${CMAKE_COMMAND} --build .. -- ${PARALLEL_BUILD}
        RESULT_VARIABLE GLFW_BUILD
        WORKING_DIRECTORY ${GLFW_ROOT}/build)

    if(GLFW_BUILD)
        message(FATAL_ERROR "Building GLFW failed: " ${GLFW_BUILD})
    endif(GLFW_BUILD)

    set(GLFW_LIBRARIES "${CMAKE_BINARY_DIR}/lib")
    set(GLFW_INCLUDE_DIRS "${CMAKE_BINARY_DIR}/include")

    find_library(GLFW_LIBRARY glfw glfw3 NO_DEFAULT_PATH HINTS "${GLFW_LIBRARIES}")

    if(GLFW_LIBRARY)
        set(GLFW_FOUND TRUE)
    endif(GLFW_LIBRARY)
endif()

message(STATUS "GLFW_LIBRARY: ${GLFW_LIBRARY}")
message(STATUS "GLFW_LIBRARIES: ${GLFW_LIBRARIES}")
message(STATUS "GLFW_INCLUDE_DIRS: ${GLFW_INCLUDE_DIRS}")
