cmake_minimum_required(VERSION 3.12)

# Set FreeImage version from Source/FreeImage.h
file(STRINGS "Source/FreeImage.h" _contents REGEX "^#define FREEIMAGE_MAJOR_VERSION.+ [0-9]+")
if(_contents)
	string(REGEX REPLACE ".*FREEIMAGE_MAJOR_VERSION[ \t]+([0-9]+).*" "\\1" FREEIMAGE_VERSION_MAJOR "${_contents}")
endif()
file(STRINGS "Source/FreeImage.h" _contents REGEX "^#define FREEIMAGE_MINOR_VERSION.+ [0-9]+")
if(_contents)
	string(REGEX REPLACE ".*FREEIMAGE_MINOR_VERSION[ \t]+([0-9]+).*" "\\1" FREEIMAGE_VERSION_MINOR "${_contents}")
endif()
file(STRINGS "Source/FreeImage.h" _contents REGEX "^#define FREEIMAGE_RELEASE_SERIAL.+ [0-9]+")
if(_contents)
	string(REGEX REPLACE ".*FREEIMAGE_RELEASE_SERIAL[ \t]+([0-9]+).*" "\\1" FREEIMAGE_RELEASE_SERIAL "${_contents}")
endif()
set(FREEIMAGE_VERSION ${FREEIMAGE_VERSION_MAJOR}.${FREEIMAGE_VERSION_MINOR}.${FREEIMAGE_RELEASE_SERIAL} )

option(ENABLE_PNG "Enable PNG support" ON)
option(ENABLE_RAW "Enable RAW support" OFF)
option(ENABLE_JPEG "Enable JPEG support" ON)
option(ENABLE_TIFF "Enable TIFF support" ON)
option(ENABLE_WEBP "Enable WebP support" OFF)
option(ENABLE_OPENJP "Enable OpenJPEG support" OFF)
option(FREEIMAGE_DYNAMIC_C_RUNTIME "If ON, Dynamically link C/C++ runtime" ON)

if(NOT FREEIMAGE_DYNAMIC_C_RUNTIME)
	# Statically link runtimes
	if(MSVC)
		set(CMAKE_C_FLAGS_DEBUG_INIT "/D_DEBUG /MTd /Zi /Ob0 /Od /RTC1")
		set(CMAKE_C_FLAGS_MINSIZEREL_INIT	 "/MT /O1 /Ob1 /D NDEBUG")
		set(CMAKE_C_FLAGS_RELEASE_INIT		"/MT /O2 /Ob2 /D NDEBUG")
		set(CMAKE_C_FLAGS_RELWITHDEBINFO_INIT "/MT /Zi /O2 /Ob1 /D NDEBUG")
		set(CMAKE_CXX_FLAGS_DEBUG_INIT "/D_DEBUG /MTd /Zi /Ob0 /Od /RTC1")
		set(CMAKE_CXX_FLAGS_MINSIZEREL_INIT	 "/MT /O1 /Ob1 /D NDEBUG")
		set(CMAKE_CXX_FLAGS_RELEASE_INIT		"/MT /O2 /Ob2 /D NDEBUG")
		set(CMAKE_CXX_FLAGS_RELWITHDEBINFO_INIT "/MT /Zi /O2 /Ob1 /D NDEBUG")
	endif()
endif(NOT FREEIMAGE_DYNAMIC_C_RUNTIME)

project (FreeImage)
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/CMakeModules)

# Check parameters
include(CheckIncludeFile)
include(CheckIncludeFileCXX)
include(CheckFunctionExists)
include(CheckTypeSize)

# Filter values through regex
#	filter_regex({INCLUDE | EXCLUDE} <regex> <listname> [items...])
#	Element will included into result list if
#		INCLUDE is specified and it matches with regex or
#		EXCLUDE is specified and it doesn't match with regex.
# Example:
#	filter_regex(INCLUDE "(a|c)" LISTOUT a b c d) => a c
#	filter_regex(EXCLUDE "(a|c)" LISTOUT a b c d) => b d
function(filter_regex _action _regex _listname)
	# check an action
	if("${_action}" STREQUAL "INCLUDE")
		set(has_include TRUE)
	elseif("${_action}" STREQUAL "EXCLUDE")
		set(has_include FALSE)
	else()
		message(FATAL_ERROR "Incorrect value for ACTION: ${_action}")
	endif()
	set(${_listname})
	foreach(element ${ARGN})
		string(REGEX MATCH ${_regex} result ${element})
		if(result)
			if(has_include)
				list(APPEND ${_listname} ${element})
			endif()
		else()
			if(NOT has_include)
				list(APPEND ${_listname} ${element})
			endif()
		endif()
	endforeach()
	# put result in parent scope variable
	set(${_listname} ${${_listname}} PARENT_SCOPE)
endfunction()

# Add the freeimage C++ source files
file(GLOB_RECURSE FREEIMAGE_SOURCES "Source/FreeImage/**/*.cpp")

# Exclude all the the optional components at first
filter_regex(EXCLUDE "FreeImage/PluginPNG\\.cpp" FREEIMAGE_SOURCES ${FREEIMAGE_SOURCES})
filter_regex(EXCLUDE "FreeImage/PluginJPEG\\.cpp" FREEIMAGE_SOURCES ${FREEIMAGE_SOURCES})
filter_regex(EXCLUDE "FreeImageToolkit/JPEGTransform\\.cpp" FREEIMAGE_SOURCES ${FREEIMAGE_SOURCES})
filter_regex(EXCLUDE "FreeImage/PluginG3\\.cpp" FREEIMAGE_SOURCES ${FREEIMAGE_SOURCES})
filter_regex(EXCLUDE "FreeImage/PluginTIFF\\.cpp" FREEIMAGE_SOURCES ${FREEIMAGE_SOURCES})
filter_regex(EXCLUDE "Metadata/XTIFF\\.cpp" FREEIMAGE_SOURCES ${FREEIMAGE_SOURCES})
filter_regex(EXCLUDE "FreeImage/PluginWebP\\.cpp" FREEIMAGE_SOURCES ${FREEIMAGE_SOURCES})
filter_regex(EXCLUDE "FreeImage/PluginRAW\\.cpp" FREEIMAGE_SOURCES ${FREEIMAGE_SOURCES})
filter_regex(EXCLUDE "FreeImage/J2KHelper\\.cpp" FREEIMAGE_SOURCES ${FREEIMAGE_SOURCES})
filter_regex(EXCLUDE "FreeImage/PluginJP2\\.cpp" FREEIMAGE_SOURCES ${FREEIMAGE_SOURCES})
filter_regex(EXCLUDE "FreeImage/PluginJ2K\\.cpp" FREEIMAGE_SOURCES ${FREEIMAGE_SOURCES})

# Find packages and add sources for optional components as necessary
FIND_PACKAGE(ZLIB REQUIRED)
set(LIBS ${ZLIB_LIBRARIES})

if(ENABLE_PNG)
	FIND_PACKAGE(PNG REQUIRED)
	set(LIBS ${LIBS} ${PNG_LIBRARIES})
	add_definitions(-DENABLE_PNG)
	set(FREEIMAGE_SOURCES ${FREEIMAGE_SOURCES} Source/FreeImage/PluginPNG.cpp)
endif()

if(ENABLE_JPEG)
	FIND_PACKAGE(JPEG REQUIRED)
	set(LIBS ${LIBS} ${JPEG_LIBRARIES})
	add_definitions(-DENABLE_JPEG)
	set(FREEIMAGE_SOURCES ${FREEIMAGE_SOURCES} Source/FreeImage/PluginJPEG.cpp)
	set(FREEIMAGE_SOURCES ${FREEIMAGE_SOURCES} Source/FreeImageToolkit/JPEGTransform.cpp)
endif()

if(ENABLE_TIFF)
	FIND_PACKAGE(TIFF REQUIRED)
	set(LIBS ${LIBS} ${TIFF_LIBRARIES})
	add_definitions(-DENABLE_TIFF)
	set(FREEIMAGE_SOURCES ${FREEIMAGE_SOURCES} Source/FreeImage/PluginG3.cpp)
	set(FREEIMAGE_SOURCES ${FREEIMAGE_SOURCES} Source/FreeImage/PluginTIFF.cpp)
	set(FREEIMAGE_SOURCES ${FREEIMAGE_SOURCES} Source/Metadata/XTIFF.cpp)
endif()

if(ENABLE_WEBP)
	find_path(WEBP_INCLUDE_DIR mux_types.h)
	find_library(WEBP_LIBRARIES NAMES webp libwebp)
	set(LIBS ${LIBS} ${WEBP_LIBRARIES})
	find_library(WEBP_MUX_LIBRARIES NAMES webpmux libwebpmux)
	set(LIBS ${LIBS} ${WEBP_MUX_LIBRARIES})
	add_definitions(-DENABLE_WEBP)
	set(FREEIMAGE_SOURCES ${FREEIMAGE_SOURCES} Source/FreeImage/PluginWebP.cpp)
endif()

if(ENABLE_RAW)
	find_path(RAW_INCLUDE_DIR libraw.h)
	find_library(RAW_LIBRARIES NAMES raw libraw)
	set(LIBS ${LIBS} ${RAW_LIBRARIES})
	set(FREEIMAGE_SOURCES ${FREEIMAGE_SOURCES} Source/FreeImage/PluginRAW.cpp)
endif()

if(ENABLE_OPENJP)
	find_path(OPENJP_INCLUDE_DIR openjpeg.h)
	find_library(OPENJP_LIBRARIES NAMES openjp2 libopenjp2)
	set(LIBS ${LIBS} ${OPENJP_LIBRARIES})
	set(FREEIMAGE_SOURCES ${FREEIMAGE_SOURCES} Source/FreeImage/J2KHelper.cpp)
	set(FREEIMAGE_SOURCES ${FREEIMAGE_SOURCES} Source/FreeImage/PluginJP2.cpp)
	set(FREEIMAGE_SOURCES ${FREEIMAGE_SOURCES} Source/FreeImage/PluginJ2K.cpp)
endif()

if(MSVC)
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

# Set libraries for static or DLL build
if(FREEIMAGE_BUILD_STATIC)
	set(FREEIMAGE_BUILD_FLAGS -DFREEIMAGE_LIB)
else(FREEIMAGE_BUILD_STATIC)
	set(FREEIMAGE_LIBRARIES FreeImage)
	set(FREEIMAGE_BUILD_DLL ON)
endif(FREEIMAGE_BUILD_STATIC)

# Set default output dirs
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(FREEIMAGE_LIBRARY_DIRS ${CMAKE_BINARY_DIR}/Lib)
set(FREEIMAGE_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/Source)

include_directories(${ZLIB_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Source)

if(FREEIMAGE_BUILD_STATIC)
	add_definitions(-DFREEIMAGE_LIB -DLIBRAW_NODLL -DOPJ_STATIC)
	add_library(FreeImage_static STATIC ${FREEIMAGE_SOURCES})
	set_target_properties(FreeImage_static PROPERTIES OUTPUT_NAME "freeimage")
	install(TARGETS FreeImage_static DESTINATION lib)
endif()

if(FREEIMAGE_BUILD_DLL)
	add_definitions(-DFREEIMAGE_EXPORTS -DLIBRAW_NODLL -DOPJ_STATIC)
	add_library(FreeImage SHARED ${FREEIMAGE_SOURCES})
	target_link_libraries(FreeImage ${LIBS})
	if(MINGW)
		target_link_libraries(FreeImage ws2_32)
	endif()
	set_target_properties(FreeImage PROPERTIES OUTPUT_NAME "freeimage")
	install(TARGETS FreeImage DESTINATION lib)
endif()

install(FILES "Source/FreeImage.h" DESTINATION include)

# if(ENABLE_TESTS)
#	ENABLE_TESTING()
#	add_subdirectory(TestAPI)
# endif()
