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

function(_get_lib_location_info dep out out_a out_a_all)
  set(location_prop "IMPORTED_LOCATION_${CMAKE_BUILD_TYPE}")
  string(TOUPPER "${location_prop}" location_prop)
  get_target_property(loc ${dep} ${location_prop})
  if(loc)
    string(REPLACE "\.a" "\.a\.all" loc_all "${loc}")
  else()
    set(loc "$<TARGET_FILE:${dep}>")
    set(loc_all "$<TARGET_FILE:${dep}>.all")
  endif()
  string(REPLACE "::" "_" dep_ "${dep}")
  set(${out} "${dep_}" PARENT_SCOPE)
  set(${out_a} "${loc}" PARENT_SCOPE)
  set(${out_a_all} "${loc_all}" PARENT_SCOPE)
endfunction()

function(_generate_arch_specific_custom_target)

  set(options "")
  set(oneValueArgs "NAME" "ARCH")
  set(multiValueArgs "DEPENDENCIES")
  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(arch "${__ARCH}")
  set(name "${__NAME}")
  set(deps "${__DEPENDENCIES}")
  set(name_all "${name}_all")
  set(name_arch "${name}_${arch}")

  set(nv_prune_commands)
  set(do_rename_commands)
  set(un_rename_commands)

  if(NOT depends_on)
    set(depends_on "${name}")
  endif()

  foreach(dep IN LISTS deps)
    if(TARGET ${dep})
      _get_lib_location_info(${dep} dep dep_a dep_a_all)
      if (dep_a)
        # Rename `${dep_a}` to `${dep_a_all}`
        add_custom_target("${dep}_${arch}_rename_a_to_all_a" ALL COMMAND ${CMAKE_COMMAND} -E rename "${dep_a}" "${dep_a_all}" DEPENDS "${depends_on}" VERBATIM COMMAND_EXPAND_LISTS)
        set(depends_on "${dep}_${arch}_rename_a_to_all_a")
        # Run nvprune to remove archs that aren't ${arch}
        add_custom_target("${dep}_${arch}_nvprune" ALL COMMAND nvprune -gencode=arch=compute_${arch},code=[sm_${arch}] -o "${dep_a}" "${dep_a_all}" DEPENDS "${depends_on}" VERBATIM COMMAND_EXPAND_LISTS)
        set(depends_on "${dep}_${arch}_nvprune")
      endif()
    endif()
  endforeach()

  add_custom_target("${name_arch}" ALL
    # Rename `${name}.node` to `${name_all}.node`
    COMMAND ${CMAKE_COMMAND} -E rename ${name}.node ${name_all}.node
    # Relink arch-specific `${name}.node`
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_CURRENT_BINARY_DIR} --target ${name}.node
    # Rename arch-specific `${name}.node` to `${name_arch}.node`
    COMMAND ${CMAKE_COMMAND} -E rename ${name}.node ${name_arch}.node
    # Rename `${name_all}.node` back to `${name}.node`
    COMMAND ${CMAKE_COMMAND} -E rename ${name_all}.node ${name}.node
    VERBATIM DEPENDS "${depends_on}" COMMAND_EXPAND_LISTS
  )
  set(depends_on "${name_arch}")

  foreach(dep IN LISTS deps)
    if(TARGET ${dep})
      _get_lib_location_info(${dep} dep dep_a dep_a_all)
      if (dep_a)
        # Rename `${dep_a_all}` to `${dep_a}`
        add_custom_target("${dep}_${arch}_rename_all_a_to_a" ALL COMMAND ${CMAKE_COMMAND} -E rename "${dep_a_all}" "${dep_a}" VERBATIM DEPENDS "${depends_on}" COMMAND_EXPAND_LISTS)
        set(depends_on "${dep}_${arch}_rename_all_a_to_a")
      endif()
    endif()
  endforeach()
  set(depends_on "${depends_on}" PARENT_SCOPE)
endfunction()

function(generate_arch_specific_custom_targets)
  set(options "")
  set(oneValueArgs "NAME")
  set(multiValueArgs "DEPENDENCIES")
  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(name ${__NAME})
  set(deps ${__DEPENDENCIES})
  set(depends_on "")

  get_target_property(cuda_archs ${name} CUDA_ARCHITECTURES)

  foreach(arch IN LISTS cuda_archs)
    if(arch MATCHES "^(.*)-(real|virtual)$")
      set(arch "${CMAKE_MATCH_1}")
    endif()
    _generate_arch_specific_custom_target(
      NAME "${name}" ARCH "${arch}" DEPENDENCIES ${deps})
  endforeach()
endfunction()
