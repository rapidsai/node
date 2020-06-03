include(FindGLEW)

if(NOT GLEW_FOUND OR (GLEW_VERSION VERSION_LESS ${REQUIRED_GLEW_VERSION}))

    message(STATUS "GLEW ${REQUIRED_GLEW_VERSION} not found, building from source")

    set(GLEW_ROOT "${CMAKE_BINARY_DIR}/glew")
    set(GLEW_GIT_BRANCH_NAME "glew-cmake-${REQUIRED_GLEW_VERSION}")

    set(GLEW_CMAKE_ARGS " -DONLY_LIBS=0"
                        " -Dglew-cmake_BUILD_MULTI_CONTEXT=OFF"
                        " -Dglew-cmake_BUILD_SINGLE_CONTEXT=ON"
                        " -Dglew-cmake_BUILD_SHARED=${GLEW_USE_SHARED_LIBS}"
                        " -Dglew-cmake_BUILD_STATIC=${GLEW_USE_STATIC_LIBS}")


    configure_file("${CMAKE_CURRENT_LIST_DIR}/../Templates/GLEW.CMakeLists.txt.cmake"
                   "${GLEW_ROOT}/CMakeLists.txt")

    execute_process(
        COMMAND ${CMAKE_COMMAND} -Wno-dev -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE GLEW_CONFIG
        WORKING_DIRECTORY ${GLEW_ROOT})

    if(GLEW_CONFIG)
        message(FATAL_ERROR "Configuring GLEW failed: " ${GLEW_CONFIG})
    endif(GLEW_CONFIG)

    execute_process(
        COMMAND ${CMAKE_COMMAND} --build .. -- ${PARALLEL_BUILD}
        RESULT_VARIABLE GLEW_BUILD
        WORKING_DIRECTORY ${GLEW_ROOT}/build)

    if(GLEW_BUILD)
        message(FATAL_ERROR "Building GLEW failed: " ${GLEW_BUILD})
    endif(GLEW_BUILD)

    set(GLEW_LIBRARIES "${CMAKE_BINARY_DIR}/lib")
    set(GLEW_INCLUDE_DIRS "${CMAKE_BINARY_DIR}/include")

    find_library(GLEW_LIBRARY glew NO_DEFAULT_PATH HINTS "${GLEW_LIBRARIES}")

    if(GLEW_LIBRARY)
        message(STATUS "GLEW library: ${GLEW_LIBRARY}")
        set(GLEW_FOUND TRUE)
    endif(GLEW_LIBRARY)
endif()

message(STATUS "GLEW_LIBRARY: ${GLEW_LIBRARY}")
message(STATUS "GLEW_LIBRARIES: ${GLEW_LIBRARIES}")
message(STATUS "GLEW_INCLUDE_DIRS: ${GLEW_INCLUDE_DIRS}")
