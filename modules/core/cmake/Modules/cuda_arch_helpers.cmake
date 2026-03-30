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

function(get_all_static_dependencies _target _outvar)
  set(__SEEN)
  _get_all_static_dependencies("${_target}" ${_outvar} SEEN ${__SEEN})
  set(${_outvar} ${${_outvar}} PARENT_SCOPE)
endfunction()

function(_get_all_static_dependencies _target _outvar)
  set(options "")
  set(oneValueArgs "")
  set(multiValueArgs "SEEN")
  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  while(TRUE)
    if(_target MATCHES "<LINK_ONLY:(.*[^>])")
      message(VERBOSE "get_all_static_dependencies: target=${_target} (before)")
      set(_target "${CMAKE_MATCH_1}")
      message(VERBOSE "get_all_static_dependencies: target=${_target}  (after)")
    elseif(_target MATCHES "<COMPILE_ONLY:(.*[^>])")
      message(VERBOSE "get_all_static_dependencies: target=${_target} (before)")
      set(_target "${CMAKE_MATCH_1}")
      message(VERBOSE "get_all_static_dependencies: target=${_target}  (after)")
    elseif(_target MATCHES "<TARGET_NAME_IF_EXISTS:(.*[^>])")
      message(VERBOSE "get_all_static_dependencies: target=${_target} (before)")
      set(_target "${CMAKE_MATCH_1}")
      message(VERBOSE "get_all_static_dependencies: target=${_target}  (after)")
    elseif(_target MATCHES "<LINK_LIBRARY:WHOLE_ARCHIVE,(.*[^>])")
      message(VERBOSE "get_all_static_dependencies: target=${_target} (before)")
      set(_target "${CMAKE_MATCH_1}")
      message(VERBOSE "get_all_static_dependencies: target=${_target}  (after)")
    else()
      message(VERBOSE "get_all_static_dependencies: target=${_target}")
      break()
    endif()
  endwhile()

  if(NOT TARGET ${_target})
    return()
  endif()

  get_target_property(_type ${_target} TYPE)
  message(VERBOSE "get_all_static_dependencies: ${_target} type=${_type}")

  if ("${_type}" STREQUAL "STATIC_LIBRARY")
    if(NOT ${_target} IN_LIST ${_outvar})
      list(APPEND ${_outvar} ${_target})
    endif()
  elseif(
    ("${_type}" STREQUAL "UNKNOWN_LIBRARY") AND
    ("${_target}" MATCHES "CUDA::")
  )
    if(NOT ${_target} IN_LIST ${_outvar})
      list(APPEND ${_outvar} ${_target})
    endif()
  endif()

  get_target_property(_libs ${_target} LINK_LIBRARIES)

  if(_libs)
    message(VERBOSE "get_all_static_dependencies: ${_target} LINK_LIBRARIES=${_libs}")
    foreach(_lib IN LISTS _libs)
      if(NOT ${_lib} IN_LIST __SEEN)
        list(APPEND __SEEN ${_lib})
        _get_all_static_dependencies("${_lib}" ${_outvar} SEEN ${__SEEN})
      endif()
    endforeach()
  endif()

  get_target_property(_libs ${_target} INTERFACE_LINK_LIBRARIES)

  if(_libs)
    message(VERBOSE "get_all_static_dependencies: ${_target} INTERFACE_LINK_LIBRARIES=${_libs}")
    foreach(_lib IN LISTS _libs)
      if(NOT ${_lib} IN_LIST __SEEN)
        list(APPEND __SEEN ${_lib})
        _get_all_static_dependencies("${_lib}" ${_outvar} SEEN ${__SEEN})
      endif()
    endforeach()
  endif()

  set(__SEEN ${__SEEN} PARENT_SCOPE)
  set(${_outvar} ${${_outvar}} PARENT_SCOPE)
endfunction()

function(_get_lib_location_info dep arch out_dep out_loc out_loc_arch)
  set(location_prop "IMPORTED_LOCATION_${CMAKE_BUILD_TYPE}")
  string(TOUPPER "${location_prop}" location_prop)
  get_target_property(loc ${dep} ${location_prop})
  if(loc)
    string(REPLACE "\.a" "\.a\.${arch}" loc_arch "${loc}")
  else()
    set(loc "$<TARGET_FILE:${dep}>")
    set(loc_arch "$<TARGET_FILE:${dep}>.${arch}")
  endif()
  string(REPLACE "::" "_" dep_ "${dep}")
  set(${out_dep} "${dep_}" PARENT_SCOPE)
  set(${out_loc} "${loc}" PARENT_SCOPE)
  set(${out_loc_arch} "${loc_arch}" PARENT_SCOPE)
endfunction()

function(_nvprune_static_libs_and_relink)

  set(options "")
  set(oneValueArgs "NAME")
  set(multiValueArgs "ARCHS" "DEPENDENCIES")
  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(name "${__NAME}")
  set(deps "${__DEPENDENCIES}")
  set(name_all "${name}_all")

  if(NOT depends_on)
    set(depends_on "${name}")
  endif()

  find_program(nvprune_exe NAMES nvprune REQUIRED)

  set(commands)
  foreach(arch IN LISTS __ARCHS)
    foreach(tgt IN LISTS deps)
      if(TARGET ${tgt})
        _get_lib_location_info(${tgt} ${arch} dep lib lib_arch)
        message(DEBUG "${tgt}: arch= ${arch} dep= ${dep} lib= ${lib} lib_arch= ${lib_arch}")
        # Run nvprune to remove archs that aren't ${arch} and save them at `<lib>.a.<arch>`
        add_custom_target("${dep}_${arch}_nvprune" ALL
          COMMAND sudo ${nvprune_exe} -gencode=arch=compute_${arch},code=[sm_${arch}] -o "${lib_arch}" "${lib}"
          VERBATIM
          COMMAND_EXPAND_LISTS
          DEPENDS ${depends_on} ${tgt}
          USES_TERMINAL
        )
        list(APPEND commands "${dep}_${arch}_nvprune")
      endif()
    endforeach()
  endforeach()

  message(DEBUG "commands: ${commands}")

  list(LENGTH commands ary_len)
  if(ary_len GREATER 0)
    set(depends_on ${commands})
  endif()

  set(commands)
  foreach(tgt IN LISTS deps)
    if(TARGET ${tgt})
      _get_lib_location_info(${tgt} "all" dep lib lib_all)
      if(NOT (TARGET "${dep}_rename_a_to_a_all"))
        # Rename `<lib>.a` to `<lib>.a.all`
        add_custom_target("${dep}_rename_a_to_a_all" ALL
          COMMAND sudo "${CMAKE_COMMAND}" -E rename "${lib}" "${lib_all}"
          VERBATIM
          COMMAND_EXPAND_LISTS
          DEPENDS ${depends_on}
          USES_TERMINAL
        )
      endif()
      list(APPEND commands "${dep}_rename_a_to_a_all")
    endif()
  endforeach()

  message(DEBUG "commands: ${commands}")

  list(LENGTH commands ary_len)
  if(ary_len GREATER 0)
    set(depends_on ${commands})
  endif()

  set(commands)
  foreach(arch IN LISTS __ARCHS)
    set(name_arch "${name}_${arch}")

    set(rename_deps_commands)
    foreach(tgt IN LISTS deps)
      if(TARGET ${tgt})
        _get_lib_location_info(${tgt} ${arch} dep lib lib_arch)
        # Rename `<lib>.a.<arch>` to `<lib>.a`
        add_custom_target("${dep}_rename_a_${arch}_to_a" ALL
          COMMAND sudo "${CMAKE_COMMAND}" -E rename "${lib_arch}" "${lib}"
          VERBATIM
          COMMAND_EXPAND_LISTS
          DEPENDS ${depends_on}
          USES_TERMINAL
        )
        list(APPEND rename_deps_commands "${dep}_rename_a_${arch}_to_a")
      endif()
    endforeach()

    message(DEBUG "rename_deps_commands: ${rename_deps_commands}")

    list(LENGTH rename_deps_commands ary_len)
    if(ary_len GREATER 0)

      set(depends_on ${rename_deps_commands})

      add_custom_target("${name_arch}" ALL
        # Rename `${name}.node` to `${name_all}.node`
        COMMAND sudo "${CMAKE_COMMAND}" -E rename "${name}.node" "${name_all}.node"
        # Relink arch-specific `${name}.node`
        COMMAND "${CMAKE_COMMAND}" --build "${CMAKE_CURRENT_BINARY_DIR}" --target "${name}.node"
        # Rename arch-specific `${name}.node` to `${name_arch}.node`
        COMMAND sudo "${CMAKE_COMMAND}" -E rename "${name}.node" "${name_arch}.node"
        # Rename `${name_all}.node` back to `${name}.node`
        COMMAND sudo "${CMAKE_COMMAND}" -E rename "${name_all}.node" "${name}.node"
        VERBATIM
        COMMAND_EXPAND_LISTS
        DEPENDS ${depends_on}
        BYPRODUCTS "${name_arch}.node"
        USES_TERMINAL
      )

      set(depends_on "${name_arch}")
    else()
      # Copy `${name}.node` to arch-specific `${name_arch}.node`
      add_custom_target("${name_arch}_copy" ALL
        COMMAND "${CMAKE_COMMAND}" -E copy ${name}.node ${name_arch}.node
        VERBATIM
        COMMAND_EXPAND_LISTS
        DEPENDS ${depends_on}
        BYPRODUCTS "${name_arch}.node"
        USES_TERMINAL
      )
      list(APPEND commands "${name_arch}_copy")
    endif()
  endforeach()

  message(DEBUG "commands: ${commands}")

  list(LENGTH commands ary_len)
  if(ary_len GREATER 0)
    set(depends_on ${commands})
  endif()

  set(commands)
  foreach(tgt IN LISTS deps)
    if(TARGET ${tgt})
      _get_lib_location_info(${tgt} "all" dep lib lib_all)
      if(NOT (TARGET "${dep}_rename_a_all_to_a"))
        # Rename `<lib>.a.all` to `<lib>.a`
        add_custom_target("${dep}_rename_a_all_to_a" ALL
          COMMAND sudo "${CMAKE_COMMAND}" -E rename "${lib_all}" "${lib}"
          VERBATIM
          COMMAND_EXPAND_LISTS
          DEPENDS ${depends_on}
          USES_TERMINAL
        )
      endif()
      list(APPEND commands "${dep}_rename_a_all_to_a")
    endif()
  endforeach()

  message(DEBUG "commands: ${commands}")

  list(LENGTH commands ary_len)
  if(ary_len GREATER 0)
    set(depends_on ${commands})
  endif()

  set(depends_on ${depends_on} PARENT_SCOPE)
endfunction()

function(generate_arch_specific_custom_targets)
  set(options "")
  set(oneValueArgs "NAME")
  set(multiValueArgs "")
  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(name "${__NAME}")
  set(depends_on "")

  get_all_static_dependencies("${name}" deps)
  message(STATUS "${name} static deps: ${deps}")

  get_target_property(cuda_archs "${name}" CUDA_ARCHITECTURES)

  foreach(arch IN LISTS cuda_archs)
    if(arch MATCHES "^(.*)-(real|virtual)$")
      set(arch "${CMAKE_MATCH_1}")
    endif()
    list(APPEND archs ${arch})
  endforeach()

  _nvprune_static_libs_and_relink(NAME "${name}" ARCHS ${archs} DEPENDENCIES ${deps})
endfunction()
