# Try to find Wayland on a Unix system
#
# This will define:
#
#   WAYLAND_FOUND       - True if Wayland is found
#   WAYLAND_LIBRARIES   - Link these to use Wayland
#   WAYLAND_INCLUDE_DIR - Include directory for Wayland
#   WAYLAND_DEFINITIONS - Compiler flags for using Wayland
#
# In addition the following more fine grained variables will be defined:
#
#   WAYLAND_CLIENT_FOUND  WAYLAND_CLIENT_INCLUDE_DIR  WAYLAND_CLIENT_LIBRARIES
#   WAYLAND_SERVER_FOUND  WAYLAND_SERVER_INCLUDE_DIR  WAYLAND_SERVER_LIBRARIES
#   WAYLAND_EGL_FOUND     WAYLAND_EGL_INCLUDE_DIR     WAYLAND_EGL_LIBRARIES
#
# Copyright (c) 2013 Martin Gräßlin <mgraesslin@kde.org>
#

if (NOT WIN32)
    if (WAYLAND_INCLUDE_DIR AND WAYLAND_LIBRARIES)
        # In the cache already
        set(WAYLAND_FIND_QUIETLY TRUE)
    endif()

    # Use pkg-config to get the directories and then use these values
    # in the find_path() and find_library() calls
    find_package(PkgConfig)
    PKG_CHECK_MODULES(PKG_WAYLAND QUIET wayland-client wayland-server wayland-egl wayland-cursor)

    set(WAYLAND_DEFINITIONS ${PKG_WAYLAND_CFLAGS})

    find_path(WAYLAND_CLIENT_INCLUDE_DIR  NAMES wayland-client.h HINTS ${PKG_WAYLAND_INCLUDE_DIRS})
    find_path(WAYLAND_SERVER_INCLUDE_DIR  NAMES wayland-server.h HINTS ${PKG_WAYLAND_INCLUDE_DIRS})
    find_path(WAYLAND_EGL_INCLUDE_DIR     NAMES wayland-egl.h    HINTS ${PKG_WAYLAND_INCLUDE_DIRS})
    find_path(WAYLAND_CURSOR_INCLUDE_DIR  NAMES wayland-cursor.h HINTS ${PKG_WAYLAND_INCLUDE_DIRS})

    find_library(WAYLAND_CLIENT_LIBRARIES NAMES wayland-client   HINTS ${PKG_WAYLAND_LIBRARY_DIRS})
    find_library(WAYLAND_SERVER_LIBRARIES NAMES wayland-server   HINTS ${PKG_WAYLAND_LIBRARY_DIRS})
    find_library(WAYLAND_EGL_LIBRARIES    NAMES wayland-egl      HINTS ${PKG_WAYLAND_LIBRARY_DIRS})
    find_library(WAYLAND_CURSOR_LIBRARIES NAMES wayland-cursor   HINTS ${PKG_WAYLAND_LIBRARY_DIRS})

    set(WAYLAND_INCLUDE_DIR ${WAYLAND_CLIENT_INCLUDE_DIR} ${WAYLAND_SERVER_INCLUDE_DIR} ${WAYLAND_EGL_INCLUDE_DIR} ${WAYLAND_CURSOR_INCLUDE_DIR})

    set(WAYLAND_LIBRARIES ${WAYLAND_CLIENT_LIBRARIES} ${WAYLAND_SERVER_LIBRARIES} ${WAYLAND_EGL_LIBRARIES} ${WAYLAND_CURSOR_LIBRARIES})

    list(REMOVE_DUPLICATES WAYLAND_INCLUDE_DIR)

    include(FindPackageHandleStandardArgs)

    find_package_handle_standard_args(WAYLAND_CLIENT  DEFAULT_MSG  WAYLAND_CLIENT_LIBRARIES  WAYLAND_CLIENT_INCLUDE_DIR)
    find_package_handle_standard_args(WAYLAND_SERVER  DEFAULT_MSG  WAYLAND_SERVER_LIBRARIES  WAYLAND_SERVER_INCLUDE_DIR)
    find_package_handle_standard_args(WAYLAND_EGL     DEFAULT_MSG  WAYLAND_EGL_LIBRARIES     WAYLAND_EGL_INCLUDE_DIR)
    find_package_handle_standard_args(WAYLAND_CURSOR  DEFAULT_MSG  WAYLAND_CURSOR_LIBRARIES  WAYLAND_CURSOR_INCLUDE_DIR)
    find_package_handle_standard_args(WAYLAND         DEFAULT_MSG  WAYLAND_LIBRARIES         WAYLAND_INCLUDE_DIR)

    mark_as_advanced(
            WAYLAND_INCLUDE_DIR         WAYLAND_LIBRARIES
            WAYLAND_CLIENT_INCLUDE_DIR  WAYLAND_CLIENT_LIBRARIES
            WAYLAND_SERVER_INCLUDE_DIR  WAYLAND_SERVER_LIBRARIES
            WAYLAND_EGL_INCLUDE_DIR     WAYLAND_EGL_LIBRARIES
            WAYLAND_CURSOR_INCLUDE_DIR  WAYLAND_CURSOR_LIBRARIES
    )
endif()
