#=============================================================================
# Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

function(find_and_configure_glfw LIB_NAME VERSION REPO TAG USE_SHARED_LIBS USE_WAYLAND USE_EGLHEADLESS)
    include(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/get_cpm.cmake)

    _set_package_dir_if_exists(${LIB_NAME} ${LIB_NAME})

    CPMFindPackage(NAME ${LIB_NAME}
        VERSION         ${VERSION}
        GIT_REPOSITORY  ${REPO}
        GIT_TAG         ${TAG}
        GIT_SHALLOW     TRUE
        GIT_CONFIG      "advice.detachedhead=false"
        OPTIONS         "GLFW_INSTALL OFF"
                        "GLFW_BUILD_DOCS OFF"
                        "GLFW_BUILD_TESTS OFF"
                        "GLFW_BUILD_EXAMPLES OFF"
                        "BUILD_SHARED_LIBS ${USE_SHARED_LIBS}"
                        "GLFW_USE_WAYLAND ${USE_WAYLAND}"
                        "GLFW_USE_EGLHEADLESS ${USE_EGLHEADLESS}"
    )

    set(${LIB_NAME}_VERSION "${${LIB_NAME}_VERSION}" PARENT_SCOPE)
endfunction()
