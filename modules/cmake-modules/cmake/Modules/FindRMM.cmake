#
# Find RMM
#
# Try to find RMM library.
# This module defines the following variables:
# - RMM_INCLUDE_DIRS
# - RMM_LIBRARIES
# - RMM_FOUND
#
# The following variables can be set as arguments for the module.
# - RMM_ROOT_DIR : Root library directory of RMM
# - RMM_USE_STATIC_LIBS : Specifies to use static version of RMM library (Windows only)
#

# Additional modules
include(FindPackageHandleStandardArgs)

find_package(RMM CONFIG QUIET)

if(RMM_FOUND)
    find_package_handle_standard_args(RMM DEFAULT_MSG RMM_CONFIG)
    return()
endif()

if (WIN32)
	# Find include files
    # PATHS $ENV{PROGRAMFILES}/include ${RMM_ROOT_DIR}/include
	find_path(RMM_INCLUDE_DIR rmm/rmm.h DOC "The rmm/rmm.h parent dir")
	# Use rmm.lib for static library
	if (RMM_USE_STATIC_LIBS)
        set(RMM_LIBRARY_NAME rmm)
	else()
        set(RMM_LIBRARY_NAME rmmdll)
	endif()
	# Find library files
    # PATHS $ENV{PROGRAMFILES}/lib ${RMM_ROOT_DIR}/lib
    find_library(RMM_LIBRARY ${RMM_LIBRARY_NAME} DOC "The RMM library")
	unset(RMM_LIBRARY_NAME)
else()
	# Find include files
    # PATHS /usr/include /usr/local/include /sw/include /opt/local/include
	find_path(RMM_INCLUDE_DIR rmm/rmm.h DOC "The rmm/rmm.h parent dir")
	# Find library files. Try to use static libraries
    # PATHS /usr/lib64 /usr/lib /usr/local/lib64 /usr/local/lib /sw/lib /opt/local/lib ${RMM_ROOT_DIR}/lib
	find_library(RMM_LIBRARY NAMES rmm PATH_SUFFIXES lib lib64 DOC "The RMM library")
endif()

# Read version from GL/glew.h file
if(EXISTS "${RMM_INCLUDE_DIR}/rmm/rmm.h")
    file(STRINGS "${RMM_INCLUDE_DIR}/rmm/rmm.h" _contents REGEX "^#define RMM_VERSION_.+ [0-9]+")
    if(_contents)
        string(REGEX REPLACE ".*VERSION_MAJOR[ \t]+([0-9]+).*" "\\1" RMM_VERSION_MAJOR "${_contents}")
        string(REGEX REPLACE ".*VERSION_MINOR[ \t]+([0-9]+).*" "\\1" RMM_VERSION_MINOR "${_contents}")
        string(REGEX REPLACE ".*VERSION_REVISION[ \t]+([0-9]+).*" "\\1" RMM_VERSION_REVISION "${_contents}")
        set(RMM_VERSION "${RMM_VERSION_MAJOR}.${RMM_VERSION_MINOR}.${RMM_VERSION_REVISION}")
    endif()
else()
    message(STATUS "FindRMM: could not find RMM ${RMM_INCLUDE_DIR}/rmm/rmm.h")
    return()
endif()

# Define RMM_LIBRARIES and RMM_INCLUDE_DIRS
set(RMM_LIBRARIES ${RMM_LIBRARY})
set(RMM_INCLUDE_DIRS ${RMM_INCLUDE_DIR})

# Handle REQUIRED argument, define *_FOUND variable
find_package_handle_standard_args(RMM
                                  FOUND_VAR RMM_FOUND
                                  REQUIRED_VARS RMM_INCLUDE_DIRS RMM_LIBRARIES
                                  VERSION_VAR RMM_VERSION)

if(NOT RMM_FOUND)
    message(STATUS "FindRMM: could not find RMM library.")
    return()
endif()

if(RMM_VERSION VERSION_LESS ${REQUIRED_RMM_VERSION})
    set(RMM_FOUND FALSE)
    unset(RMM_LIBRARIES)
    unset(RMM_INCLUDE_DIRS)
    message(STATUS "FindRMM: found older RMM version")
    return()
endif()

if(RMM_USE_STATIC_LIBS)
    set(RMM_LIBRARY_LINK STATIC IMPORTED)
else()
    set(RMM_LIBRARY_LINK SHARED IMPORTED)
endif(RMM_USE_STATIC_LIBS)

add_library(rmm ${RMM_LIBRARY_LINK})
set_target_properties(rmm PROPERTIES
    IMPORTED_LOCATION "${RMM_LIBRARY}"
    INTERFACE_LINK_LIBRARIES OpenGL::GL
    INTERFACE_INCLUDE_DIRECTORIES "${RMM_INCLUDE_DIRS}")

# Hide some variables
mark_as_advanced(RMM_INCLUDE_DIR RMM_LIBRARY_LINK RMM_LIBRARY)
