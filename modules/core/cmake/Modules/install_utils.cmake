#=============================================================================
# Copyright (c) 2022-2026, NVIDIA CORPORATION.
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

set(_INSTALL_UTILS_DIR "${CMAKE_CURRENT_LIST_DIR}")

macro(find_packages_and_generate_exports)

  set(options "")
  set(oneValueArgs NAME)
  set(multiValueArgs CPM_DEPS NPM_DEPS MODULES)

  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  include("${_INSTALL_UTILS_DIR}/get_cpm.cmake")

  set(_find_pkg)
  set(_configure_args -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE})

  foreach(_module IN LISTS __NPM_DEPS)
    string(REGEX REPLACE "^@" "" _module "${_module}")
    string(REGEX REPLACE "/" "_" _target "${_module}")

    string(REGEX REPLACE "^rapidsai_" "" _dir "${_target}")
    set(_path "build/${CMAKE_BUILD_TYPE}/cmake/${_target}")

    execute_process(
      COMMAND node -p "require('path').dirname(require.resolve('@${_module}'))"
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      OUTPUT_STRIP_TRAILING_WHITESPACE
      OUTPUT_VARIABLE _module_path
      COMMAND_ERROR_IS_FATAL ANY
    )

    set(${_target}_ROOT "${_module_path}/${_path}")
    message(STATUS "${_target}_ROOT=${${_target}_ROOT}")
    find_package(${_target} $ENV{npm_package_version} REQUIRED)

    list(APPEND _configure_args -D${_target}_ROOT=${${_target}_ROOT})
    list(APPEND _find_pkg "find_package(${_target} $ENV{npm_package_version} REQUIRED)")

    rapids_export_package(BUILD ${_target} ${__NAME}-exports GLOBAL_TARGETS rapidsai::${_target})
    rapids_export_package(INSTALL ${_target} ${__NAME}-exports GLOBAL_TARGETS rapidsai::${_target})

    rapids_export_find_package_root(BUILD ${_target} "${${_target}_ROOT}" EXPORT_SET ${__NAME}-exports)
    rapids_export_find_package_root(INSTALL ${_target} "\${PACKAGE_PREFIX_DIR}/../../../${_dir}/${_path}" EXPORT_SET ${__NAME}-exports)
  endforeach()

  list(TRANSFORM __MODULES PREPEND [=[include("]=])
  list(TRANSFORM __MODULES APPEND [=[")]=])
  string(JOIN "\n" __MODULES ${__MODULES})
  string(JOIN "\n" _find_pkg ${_find_pkg})

  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/build/${CMAKE_BUILD_TYPE}/${__NAME}_lib.a")

    file(CONFIGURE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/_lib-src/stub.c" CONTENT "")

    string(
      JOIN "\n" _lib_cmakelists_txt
      [=[
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
cmake_minimum_required(VERSION 4.2.3 FATAL_ERROR)

include("${rapidsai-modules}/cmake_policies.cmake")

project(${__NAME}_lib VERSION 1.0.0 LANGUAGES C CXX)

include("${rapidsai-modules}/ConfigureCXX.cmake")
include("${rapidsai-modules}/ConfigureCUDA.cmake")
include("${rapidsai-modules}/ConfigureNapi.cmake")
include("${rapidsai-modules}/install_utils.cmake")

add_library(${__NAME}_lib STATIC "${CMAKE_CURRENT_BINARY_DIR}/_lib-src/stub.c")
add_library(rapidsai::${__NAME}_lib ALIAS ${__NAME}_lib)

set_target_properties(${__NAME}_lib
    PROPERTIES PREFIX                              ""
               BUILD_RPATH                         "\$ORIGIN"
               INSTALL_RPATH                       "\$ORIGIN"
               CXX_STANDARD                        20
               CXX_STANDARD_REQUIRED               ON
               CUDA_STANDARD                       20
               CUDA_STANDARD_REQUIRED              ON
               NO_SYSTEM_FROM_IMPORTED             ON
               POSITION_INDEPENDENT_CODE           ON
               INTERFACE_POSITION_INDEPENDENT_CODE ON
)
]=]
      "${_find_pkg}"
      "${__MODULES}"
      [=[
if(NOT "${__CPM_DEPS}" STREQUAL "")
  target_link_libraries(${__NAME}_lib PUBLIC ${__CPM_DEPS})
endif()

generate_install_rules(NAME ${__NAME}_lib GLOBAL_TARGETS ${__NAME}_lib)
]=]
    )

    file(CONFIGURE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/_lib-src/CMakeLists.txt" CONTENT "${_lib_cmakelists_txt}")

    execute_process(
      COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/_lib-src ${CMAKE_CURRENT_BINARY_DIR}/_lib-bld
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      COMMAND_ECHO STDERR
      COMMAND_ERROR_IS_FATAL ANY
    )

    list(PREPEND _configure_args -G${CMAKE_GENERATOR})
    list(PREPEND _configure_args -S${CMAKE_CURRENT_BINARY_DIR}/_lib-src)
    list(PREPEND _configure_args -B${CMAKE_CURRENT_BINARY_DIR}/_lib-bld)

    execute_process(
      COMMAND ${CMAKE_COMMAND} ${_configure_args}
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      COMMAND_ECHO STDERR
      COMMAND_ERROR_IS_FATAL ANY
    )

    unset(ENV{SCCACHE_NO_DIST_COMPILE})

    execute_process(
      COMMAND ${CMAKE_COMMAND} --build "${CMAKE_CURRENT_BINARY_DIR}/_lib-bld/" --parallel $ENV{CMAKE_BUILD_PARALLEL_LEVEL}
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      COMMAND_ECHO STDERR
      COMMAND_ERROR_IS_FATAL ANY
    )

    set(ENV{SCCACHE_NO_DIST_COMPILE} "1")

    execute_process(
      COMMAND ${CMAKE_COMMAND} --install "${CMAKE_CURRENT_BINARY_DIR}/_lib-bld/" --prefix "${CMAKE_CURRENT_SOURCE_DIR}/build/${CMAKE_BUILD_TYPE}"
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      COMMAND_ECHO STDERR
      COMMAND_ERROR_IS_FATAL ANY
    )
  endif()

  list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/build/${CMAKE_BUILD_TYPE}")

  find_package(${__NAME}_lib REQUIRED)
  cmake_language(EVAL CODE "${__MODULES}")

  unset(_dir)
  unset(_path)
  unset(__NAME)
  unset(_module)
  unset(_target)
  unset(__MODULES)
  unset(__CPM_DEPS)
  unset(__NPM_DEPS)
  unset(_find_pkg)
  unset(_configure_args)
  unset(_lib_cmakelists_txt)
  unset(_node_module)
  unset(_module_path)
endmacro()

function(generate_install_rules)

  set(options INSTALL_TOP_LEVEL)
  set(oneValueArgs NAME BUILD_CODE_BLOCK INSTALL_CODE_BLOCK)
  set(multiValueArgs GLOBAL_TARGETS CUDA_ARCHITECTURES)

  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(libs)
  set(name "${__NAME}")
  set(targets ${__GLOBAL_TARGETS})
  set(cuda_archs ${__CUDA_ARCHITECTURES})
  set(build_code_block "")
  if(__BUILD_CODE_BLOCK AND (DEFINED ${__BUILD_CODE_BLOCK}))
    set(build_code_block "${${__BUILD_CODE_BLOCK}}")
  endif()
  set(install_code_block "")
  if(__INSTALL_CODE_BLOCK AND (DEFINED ${__INSTALL_CODE_BLOCK}))
    set(install_code_block "${${__INSTALL_CODE_BLOCK}}")
  endif()

  if(NOT targets)
    list(APPEND targets ${name})
  endif()

  list(LENGTH cuda_archs cuda_archs_len)
  if(cuda_archs_len GREATER 0)
    include("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/cuda_arch_helpers.cmake")
    generate_arch_specific_custom_targets(NAME "${name}")

    foreach(arch IN LISTS cuda_archs)
      if(arch MATCHES "^(.*)-(real|virtual)$")
        set(arch "${CMAKE_MATCH_1}")
      endif()
      list(APPEND libs "${CMAKE_CURRENT_BINARY_DIR}/${name}_${arch}.node")
    endforeach()
  endif()

  include(CPack)
  include(GNUInstallDirs)
  include("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/get_cpm.cmake")

  if(__INSTALL_TOP_LEVEL)
    set(CMAKE_INSTALL_LIBDIR ".")
  endif()
  rapids_cmake_install_lib_dir(lib_dir)
  set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME ${name})

  # install target
  install(TARGETS ${targets}
          DESTINATION ${lib_dir}
          EXPORT ${name}-exports)

  if(libs)
    install(FILES ${libs} TYPE LIB)
  endif()

  rapids_export(BUILD ${name}
    EXPORT_SET ${name}-exports
    GLOBAL_TARGETS ${targets}
    LANGUAGES CUDA
    NAMESPACE rapidsai::
    FINAL_CODE_BLOCK build_code_block
  )

  rapids_export(INSTALL ${name}
    EXPORT_SET ${name}-exports
    GLOBAL_TARGETS ${targets}
    NAMESPACE rapidsai::
    FINAL_CODE_BLOCK install_code_block
  )
endfunction()
