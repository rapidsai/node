#=============================================================================
# Copyright (c) 2020-2026, NVIDIA CORPORATION.
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

function(find_and_configure_glew)

  set(options "")
  set(oneValueArgs VERSION USE_STATIC EXPORT_SET)
  set(multiValueArgs "")

  cmake_parse_arguments(PKG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if(PKG_USE_STATIC)
      set(GLEW_USE_STATIC_LIBS ON)
      set(GLEW_USE_SHARED_LIBS OFF)
      set(GLEW_LIBRARY libglew_static)
  else()
      set(GLEW_USE_SHARED_LIBS ON)
      set(GLEW_USE_STATIC_LIBS OFF)
      set(GLEW_LIBRARY libglew_shared)
  endif()

  include(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/get_cpm.cmake)

  include(GNUInstallDirs)
  # set(CMAKE_INSTALL_LIBDIR ".")
  rapids_cmake_install_lib_dir(lib_dir)

  if(NOT TARGET ${GLEW_LIBRARY})
    rapids_cpm_find(glew ${PKG_VERSION}
      GLOBAL_TARGETS     ${GLEW_LIBRARY}
      BUILD_EXPORT_SET   ${PKG_EXPORT_SET}
      INSTALL_EXPORT_SET ${PKG_EXPORT_SET}
      CPM_ARGS
        GIT_REPOSITORY   https://github.com/Perlmint/glew-cmake.git
        GIT_TAG          glew-cmake-${PKG_VERSION}
        CUSTOM_CACHE_KEY glew-cmake-${PKG_VERSION}
        GIT_SHALLOW      TRUE
        GIT_CONFIG       "advice.detachedhead=false"
        OPTIONS          "ONLY_LIBS 0"
                         # Ignore glew's missing VERSION
                         "CMAKE_POLICY_DEFAULT_CMP0048 NEW"
                         "glew-cmake_BUILD_MULTI_CONTEXT OFF"
                         "glew-cmake_BUILD_SINGLE_CONTEXT ON"
                         "glew-cmake_BUILD_SHARED ${GLEW_USE_SHARED_LIBS}"
                         "glew-cmake_BUILD_STATIC ${GLEW_USE_STATIC_LIBS}"
    )
  endif()

  if(glew_ADDED)
    install(
      TARGETS ${GLEW_LIBRARY}
      DESTINATION ${lib_dir}
      EXPORT glew-exports
    )
    rapids_export(
      BUILD glew
      VERSION ${PKG_VERSION}
      EXPORT_SET glew-exports
      GLOBAL_TARGETS ${GLEW_LIBRARY}
    )
    rapids_export(
      INSTALL glew
      VERSION ${PKG_VERSION}
      EXPORT_SET glew-exports
      GLOBAL_TARGETS ${GLEW_LIBRARY}
    )

    target_compile_definitions(${GLEW_LIBRARY} PUBLIC GLEW_EGL)

    if(PKG_USE_STATIC)
        set_target_properties(${GLEW_LIBRARY}
            PROPERTIES POSITION_INDEPENDENT_CODE           ON
                       INTERFACE_POSITION_INDEPENDENT_CODE ON)
    endif()
  endif()

  rapids_export_package(BUILD glew ${PKG_EXPORT_SET})
  rapids_export_package(INSTALL glew ${PKG_EXPORT_SET})
  include("${rapids-cmake-dir}/export/find_package_root.cmake")
  rapids_export_find_package_root(BUILD glew "\${PACKAGE_PREFIX_DIR}/lib/cmake/glew" EXPORT_SET ${PKG_EXPORT_SET})
  rapids_export_find_package_root(INSTALL glew "\${PACKAGE_PREFIX_DIR}/lib/cmake/glew" EXPORT_SET ${PKG_EXPORT_SET})

  set(glew_VERSION "${glew_VERSION}" PARENT_SCOPE)
  set(GLEW_LIBRARY "${GLEW_LIBRARY}" PARENT_SCOPE)
endfunction()

find_and_configure_glew(
    VERSION 2.1.0
    USE_STATIC OFF # ${NODE_RAPIDS_WEBGL_STATIC_LINK}
    EXPORT_SET ${PROJECT_NAME}-exports
)
