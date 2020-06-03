#
# Find GLFW
#
# Try to find GLFW library.
# This module defines the following variables:
# - GLFW_INCLUDE_DIRS
# - GLFW_LIBRARIES
# - GLFW_FOUND
#
# The following variables can be set as arguments for the module.
# - GLFW_ROOT_DIR : Root library directory of GLFW
# - GLFW_USE_STATIC_LIBS : Specifies to use static version of GLFW library (Windows only)
#
# References:
# - https://github.com/progschj/OpenGL-Examples/blob/master/cmake_modules/FindGLFW.cmake
# - https://bitbucket.org/Ident8/cegui-mk2-opengl3-renderer/src/befd47200265/cmake/FindGLFW.cmake
#

# Additional modules
include(FindPackageHandleStandardArgs)

find_package(GLFW CONFIG QUIET)

if(GLFW_FOUND)
    find_package_handle_standard_args(GLFW DEFAULT_MSG GLFW_CONFIG)
    return()
endif()

if (WIN32)
	# Find include files
    # PATHS $ENV{PROGRAMFILES}/include ${GLFW_ROOT_DIR}/include
	find_path(GLFW_INCLUDE_DIR GLFW/glfw3.h DOC "The GLFW/glfw3.h parent dir")
	# Use glfw3.lib for static library
	if (GLFW_USE_STATIC_LIBS)
        set(GLFW_LIBRARY_NAME glfw)
	else()
        set(GLFW_LIBRARY_NAME glfwdll)
	endif()
	# Find library files
    # PATHS $ENV{PROGRAMFILES}/lib ${GLFW_ROOT_DIR}/lib
    find_library(GLFW_LIBRARY ${GLFW_LIBRARY_NAME} DOC "The GLFW library")
	unset(GLFW_LIBRARY_NAME)
else()
	# Find include files
    # PATHS /usr/include /usr/local/include /sw/include /opt/local/include
	find_path(GLFW_INCLUDE_DIR GLFW/glfw3.h DOC "The GLFW/glfw3.h parent dir")
	# Find library files. Try to use static libraries
    # PATHS /usr/lib64 /usr/lib /usr/local/lib64 /usr/local/lib /sw/lib /opt/local/lib ${GLFW_ROOT_DIR}/lib
	find_library(GLFW_LIBRARY NAMES glfw PATH_SUFFIXES lib lib64 DOC "The GLFW library")
endif()

# Read version from GL/glew.h file
if(EXISTS "${GLFW_INCLUDE_DIR}/GLFW/glfw3.h")
    file(STRINGS "${GLFW_INCLUDE_DIR}/GLFW/glfw3.h" _contents REGEX "^#define GLFW_VERSION_.+ [0-9]+")
    if(_contents)
        string(REGEX REPLACE ".*VERSION_MAJOR[ \t]+([0-9]+).*" "\\1" GLFW_VERSION_MAJOR "${_contents}")
        string(REGEX REPLACE ".*VERSION_MINOR[ \t]+([0-9]+).*" "\\1" GLFW_VERSION_MINOR "${_contents}")
        string(REGEX REPLACE ".*VERSION_REVISION[ \t]+([0-9]+).*" "\\1" GLFW_VERSION_REVISION "${_contents}")
        set(GLFW_VERSION "${GLFW_VERSION_MAJOR}.${GLFW_VERSION_MINOR}.${GLFW_VERSION_REVISION}")
    endif()
else()
    message(STATUS "FindGLFW: could not find GLFW ${GLFW_INCLUDE_DIR}/GLFW/glfw3.h")
    return()
endif()

# Define GLFW_LIBRARIES and GLFW_INCLUDE_DIRS
set(GLFW_LIBRARIES ${GLFW_LIBRARY})
set(GLFW_INCLUDE_DIRS ${GLFW_INCLUDE_DIR})

find_package_handle_standard_args(GLFW
                                  FOUND_VAR GLFW_FOUND
                                  REQUIRED_VARS GLFW_INCLUDE_DIRS GLFW_LIBRARIES
                                  VERSION_VAR GLFW_VERSION)

# Handle REQUIRED argument, define *_FOUND variable
# find_package_handle_standard_args(GLFW DEFAULT_MSG GLFW_INCLUDE_DIR GLFW_LIBRARY)

if(NOT GLFW_FOUND)
    message(STATUS "FindGLFW: could not find GLFW library.")
    return()
endif()

if(GLFW_VERSION VERSION_LESS ${REQUIRED_GLFW_VERSION})
    set(GLFW_FOUND FALSE)
    unset(GLFW_LIBRARIES)
    unset(GLFW_INCLUDE_DIRS)
    message(STATUS "FindGLFW: found older GLFW version")
    return()
endif()

if(GLFW_USE_STATIC_LIBS)
    set(GLFW_LIBRARY_LINK STATIC IMPORTED)
else()
    set(GLFW_LIBRARY_LINK SHARED IMPORTED)
endif(GLFW_USE_STATIC_LIBS)

add_library(glfw ${GLFW_LIBRARY_LINK})
set_target_properties(glfw PROPERTIES
    IMPORTED_LOCATION "${GLFW_LIBRARY}"
    INTERFACE_LINK_LIBRARIES OpenGL::GL
    INTERFACE_INCLUDE_DIRECTORIES "${GLFW_INCLUDE_DIRS}")

# Hide some variables
mark_as_advanced(GLFW_INCLUDE_DIR GLFW_LIBRARY_LINK GLFW_LIBRARY)
