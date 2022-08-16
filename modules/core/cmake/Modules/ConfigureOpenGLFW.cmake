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
include_guard(GLOBAL)

function(find_and_configure_glfw)

    set(options "")
    set(oneValueArgs VARIANT VERSION GIT_REPO GIT_TAG USE_SHARED_LIBS USE_WAYLAND USE_EGLHEADLESS EXPORT_SET)
    set(multiValueArgs "")

    cmake_parse_arguments(PKG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    include(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/get_cpm.cmake)

    set(GLFW_LIBRARY "glfw_${PKG_VARIANT}")

    _set_package_dir_if_exists(${GLFW_LIBRARY} ${GLFW_LIBRARY})

    rapids_cpm_find(${GLFW_LIBRARY} ${PKG_VERSION}
      GLOBAL_TARGETS     glfw3_${PKG_VARIANT}
                         glfw::${PKG_VARIANT}
      BUILD_EXPORT_SET   ${PKG_EXPORT_SET}
      CPM_ARGS
        GIT_REPOSITORY   ${PKG_GIT_REPO}
        GIT_TAG          ${PKG_GIT_TAG}
        GIT_SHALLOW      TRUE
        GIT_CONFIG       "advice.detachedhead=false"
        OPTIONS          "GLFW_INSTALL OFF"
                         "GLFW_BUILD_DOCS OFF"
                         "GLFW_BUILD_TESTS OFF"
                         "GLFW_BUILD_EXAMPLES OFF"
                         "BUILD_SHARED_LIBS ${PKG_USE_SHARED_LIBS}"
                         "GLFW_USE_WAYLAND ${PKG_USE_WAYLAND}"
                         "GLFW_USE_EGLHEADLESS ${PKG_USE_EGLHEADLESS}"
    )

    if(${GLFW_LIBRARY}_ADDED)
      install(TARGETS glfw3_${PKG_VARIANT}
              EXPORT  ${GLFW_LIBRARY}-exports)
      rapids_export(
        BUILD            ${GLFW_LIBRARY}
        VERSION          ${PKG_VERSION}
        EXPORT_SET       ${GLFW_LIBRARY}-exports
        GLOBAL_TARGETS   glfw3_${PKG_VARIANT}
                         glfw::${PKG_VARIANT}
      )
      rapids_export_package(BUILD glew ${PKG_EXPORT_SET})
      include("${rapids-cmake-dir}/export/find_package_root.cmake")
      rapids_export_find_package_root(BUILD glew [=[${CMAKE_CURRENT_LIST_DIR}]=] ${PKG_EXPORT_SET})
    endif()

    set(${GLFW_LIBRARY}_VERSION "${${GLFW_LIBRARY}_VERSION}" PARENT_SCOPE)
endfunction()
