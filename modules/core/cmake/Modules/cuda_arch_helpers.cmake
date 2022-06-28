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

function(generate_arch_specific_custom_target)

  set(options "")
  set(oneValueArgs "NAME" "ARCH")
  set(multiValueArgs "DEPENDENCIES")
  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(arch "${__ARCH}")
  set(name "${__NAME}")
  set(deps "${__DEPENDENCIES}")
  set(name_all "${name}_all")
  set(name_arch "${name}_${arch}")

  message(STATUS "arch: ${arch}")
  message(STATUS "name: ${name}")
  message(STATUS "name_all: ${name_all}")
  message(STATUS "name_arch: ${name_arch}")
  message(STATUS "deps:\n${deps}")

  set(nv_prune_commands)
  set(do_rename_commands)
  set(un_rename_commands)

  if(NOT depends_name)
    set(depends_name "${name}")
  endif()

  set(imported_location_prop "IMPORTED_LOCATION_${CMAKE_BUILD_TYPE}")
  string(TOUPPER "${imported_location_prop}" imported_location_prop)

  foreach(dep_name IN LISTS deps)
    message(STATUS "dep_name: ${dep_name}")
    get_target_property(dep_a ${dep_name} ${imported_location_prop})
    message(STATUS "${dep_name} ${imported_location_prop}: ${dep_a}")
    if (dep_a)
      string(REPLACE "::" "_" dep_name "${dep_name}")
      string(REPLACE "\.a" "\.all\.a" dep_all_a "${dep_a}")
      # Rename `${dep_a}` to `${dep_all_a}`
      add_custom_target("${dep_name}_${arch}_rename_a_to_all_a" ALL VERBATIM DEPENDS "${depends_name}" COMMAND ${CMAKE_COMMAND} -E rename "${dep_a}" "${dep_all_a}")
      set(depends_name "${dep_name}_${arch}_rename_a_to_all_a")
      # Run nvprune to remove archs that aren't ${arch}
      add_custom_target("${dep_name}_${arch}_nvprune" ALL VERBATIM DEPENDS "${depends_name}" COMMAND nvprune -gencode=arch=compute_${arch},code=[sm_${arch}] -o "${dep_a}" "${dep_all_a}")
      set(depends_name "${dep_name}_${arch}_nvprune")
    endif()
  endforeach()

  add_custom_target("${name_arch}" ALL
    VERBATIM DEPENDS "${depends_name}"
    # Rename `rapidsai_cuspatial.node` to `rapidsai_cuspatial_all.node`
    COMMAND ${CMAKE_COMMAND} -E rename ${name}.node ${name_all}.node
    # Relink arch-specific `rapidsai_cuspatial.node`
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_CURRENT_BINARY_DIR} --target ${name}.node
    # Rename arch-specific `rapidsai_cuspatial.node` to `rapidsai_cuspatial_${arch}.node`
    COMMAND ${CMAKE_COMMAND} -E rename ${name}.node ${name_arch}.node
    # Rename `rapidsai_cuspatial_all.node` back to `rapidsai_cuspatial.node`
    COMMAND ${CMAKE_COMMAND} -E rename ${name_all}.node ${name}.node
  )
  set(depends_name "${name_arch}")

  foreach(dep_name IN LISTS deps)
    message(STATUS "dep_name: ${dep_name}")
    get_target_property(dep_a ${dep_name} ${imported_location_prop})
    message(STATUS "${dep_name} ${imported_location_prop}: ${dep_a}")
    if (dep_a)
      string(REPLACE "::" "_" dep_name "${dep_name}")
      string(REPLACE "\.a" "\.all\.a" dep_all_a "${dep_a}")
      # Rename `${dep_all_a}` to `${dep_a}`
      add_custom_target("${dep_name}_${arch}_rename_all_a_to_a" ALL VERBATIM DEPENDS "${depends_name}" COMMAND ${CMAKE_COMMAND} -E rename "${dep_all_a}" "${dep_a}")
      set(depends_name "${dep_name}_${arch}_rename_all_a_to_a")
    endif()
  endforeach()
  set(depends_name "${depends_name}" PARENT_SCOPE)
endfunction()

function(generate_arch_specific_custom_targets)
  set(options "")
  set(oneValueArgs "NAME")
  set(multiValueArgs "DEPENDENCIES")
  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(name ${__NAME})
  set(deps ${__DEPENDENCIES})
  set(depends_name "")

  get_target_property(cuda_archs ${name} CUDA_ARCHITECTURES)

  foreach(arch IN LISTS cuda_archs)
    if(arch MATCHES "^(.*)-(real|virtual)$")
      set(arch "${CMAKE_MATCH_1}")
    endif()
    generate_arch_specific_custom_target(
      NAME "${name}" ARCH "${arch}" DEPENDENCIES ${deps})
  endforeach()
endfunction()
