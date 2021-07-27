#=============================================================================
# Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

include(get_cpm)

_set_package_dir_if_exists(glfw glfw)

function(find_and_configure_glfw VERSION REPO TAG)
    CPMFindPackage(NAME glfw
        VERSION         ${VERSION}
        GIT_REPOSITORY  ${REPO}
        GIT_TAG         ${TAG}
        GIT_SHALLOW     TRUE
        GIT_CONFIG      "advice.detachedhead=false"
        OPTIONS         "GLFW_INSTALL OFF"
                        "GLFW_BUILD_DOCS OFF"
                        "GLFW_BUILD_TESTS OFF"
                        "GLFW_BUILD_EXAMPLES OFF"
                        "BUILD_SHARED_LIBS ${GLFW_USE_SHARED_LIBS}"
                        "GLFW_USE_EGLHEADLESS ${GLFW_USE_EGLHEADLESS}"
    )
endfunction()

find_and_configure_glfw(${GLFW_VERSION} ${GLFW_GIT_REPOSITORY} ${GLFW_GIT_BRANCH})
