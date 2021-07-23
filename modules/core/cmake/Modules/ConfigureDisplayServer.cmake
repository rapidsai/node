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

set(DISPLAY_SERVER "")
set(DISPLAY_SERVER_LIBRARIES "")
set(DISPLAY_SERVER_INCLUDE_DIRS "")

if(LINUX)
    # Detect whether the current display server is X11 or Wayland
    if(DEFINED ENV{DISPLAY})
        set(DISPLAY_SERVER "X11")
    elseif(DEFINED ENV{WAYLAND_DISPLAY})
        set(DISPLAY_SERVER "Wayland")
    else()
        execute_process(
            COMMAND bash -c "loginctl show-session $(loginctl | grep $(whoami) | awk '{print $1}') -p Type"
            OUTPUT_VARIABLE display_server_session_type
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        string(FIND "${display_server_session_type}" "Type=x11" x11_loc)
        if ("${x11_loc}" STREQUAL "-1")
            set(DISPLAY_SERVER "Wayland")
        else()
            set(DISPLAY_SERVER "X11")
        endif()
    endif()
    message(STATUS "Detected display server: ${DISPLAY_SERVER}")
    if(DISPLAY_SERVER STREQUAL "Wayland")
        include(FindWayland)
        if(NOT WAYLAND_FOUND)
            message(FATAL_ERROR "Wayland detected but not found")
        endif()
        set(GLFW_USE_WAYLAND ON)
        set(DISPLAY_SERVER_LIBRARIES "${WAYLAND_LIBRARIES}")
        set(DISPLAY_SERVER_INCLUDE_DIRS "${WAYLAND_INCLUDE_DIRS}")
        message(STATUS "Wayland libraries: " "${WAYLAND_LIBRARIES}")
        message(STATUS "Wayland include dir: " "${WAYLAND_INCLUDE_DIRS}")
        list(APPEND NODE_RAPIDS_CMAKE_C_FLAGS -DGLFW_EXPOSE_NATIVE_WAYLAND)
        list(APPEND NODE_RAPIDS_CMAKE_CXX_FLAGS -DGLFW_EXPOSE_NATIVE_WAYLAND)
    else()
        include(FindX11)
        if(NOT X11_FOUND)
            message(FATAL_ERROR "X11 detected but not found")
        endif()
        set(GLFW_USE_WAYLAND OFF)
        set(DISPLAY_SERVER_LIBRARIES "${X11_LIBRARIES}")
        set(DISPLAY_SERVER_INCLUDE_DIRS "${X11_INCLUDE_DIR}")
        message(STATUS "X11 libraries: " "${X11_LIBRARIES}")
        message(STATUS "X11 include dir: " "${X11_INCLUDE_DIR}")
        list(APPEND NODE_RAPIDS_CMAKE_C_FLAGS -DGLFW_EXPOSE_NATIVE_X11)
        list(APPEND NODE_RAPIDS_CMAKE_CXX_FLAGS -DGLFW_EXPOSE_NATIVE_X11)
    endif()
endif(LINUX)
