#=============================================================================
# Copyright (c) 2022, NVIDIA CORPORATION.
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

function(generate_install_rules)

  set(options "")
  set(oneValueArgs NAME)
  set(multiValueArgs GLOBAL_TARGETS CUDA_ARCHITECTURES)

  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(libs)
  set(name "${__NAME}")
  set(targets ${__GLOBAL_TARGETS})
  set(cuda_archs ${__CUDA_ARCHITECTURES})

  if(NOT targets)
    list(APPEND targets ${name})
  endif()

  foreach(arch IN LISTS cuda_archs)
    if(arch MATCHES "^(.*)-(real|virtual)$")
      set(arch "${CMAKE_MATCH_1}")
    endif()
    list(APPEND libs "${CMAKE_CURRENT_BINARY_DIR}/${name}_${arch}.node")
  endforeach()

  include(CPack)
  include(GNUInstallDirs)
  include(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/get_cpm.cmake)

  rapids_cmake_install_lib_dir(lib_dir)
  set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME ${name})

  # install target
  install(TARGETS ${targets}
          DESTINATION ${lib_dir}
          EXPORT ${name}-exports)

  if(libs)
    install(FILES ${libs} TYPE LIB)
  endif()

  set(doc_string "")
  set(install_code_string "")

  rapids_export(
    BUILD ${name}
    EXPORT_SET ${name}-exports
    GLOBAL_TARGETS ${targets}
    NAMESPACE rapids::
    DOCUMENTATION doc_string
    FINAL_CODE_BLOCK install_code_string
  )

endfunction()
