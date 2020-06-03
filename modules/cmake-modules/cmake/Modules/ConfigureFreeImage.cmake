# include(FindFreeImage)

if(NOT FREEIMAGE_FOUND OR (FREEIMAGE_VERSION VERSION_LESS ${REQUIRED_FREEIMAGE_VERSION}))

    message(STATUS "FreeImage ${REQUIRED_FREEIMAGE_VERSION} not found, building from source")

    set(FREEIMAGE_ROOT "${CMAKE_BINARY_DIR}/freeimage")

    set(FREEIMAGE_CMAKE_ARGS " -DCMAKE_C_FLAGS=-w"
                             " -DCMAKE_CXX_FLAGS=-w"
                             " -DENABLE_PNG=ON"
                             " -DENABLE_RAW=ON"
                             " -DENABLE_JPEG=ON"
                             " -DENABLE_TIFF=ON"
                             " -DENABLE_WEBP=ON"
                             " -DENABLE_OPENJP=ON"
                             " -DFREEIMAGE_DYNAMIC_C_RUNTIME=ON")

    configure_file("${CMAKE_CURRENT_LIST_DIR}/../Templates/FreeImage.CMakeLists.txt.cmake"
                   "${FREEIMAGE_ROOT}/CMakeLists.txt")

    execute_process(
        COMMAND ${CMAKE_COMMAND} -Wno-dev -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE FREEIMAGE_CONFIG
        WORKING_DIRECTORY ${FREEIMAGE_ROOT})

    if(FREEIMAGE_CONFIG)
        message(FATAL_ERROR "Configuring FreeImage failed: " ${FREEIMAGE_CONFIG})
    endif(FREEIMAGE_CONFIG)

    execute_process(
        COMMAND ${CMAKE_COMMAND} --build .. -- ${PARALLEL_BUILD}
        RESULT_VARIABLE FREEIMAGE_BUILD
        WORKING_DIRECTORY ${FREEIMAGE_ROOT}/build)

    if(FREEIMAGE_BUILD)
        message(FATAL_ERROR "Building FreeImage failed: " ${FREEIMAGE_BUILD})
    endif(FREEIMAGE_BUILD)

    set(FREEIMAGE_LIBRARIES "${CMAKE_BINARY_DIR}/lib")
    set(FREEIMAGE_INCLUDE_DIRS "${CMAKE_BINARY_DIR}/include")

    find_library(FREEIMAGE_LIBRARY freeimage NO_DEFAULT_PATH HINTS "${FREEIMAGE_LIBRARIES}")

    if(FREEIMAGE_LIBRARY)
        set(FREEIMAGE_FOUND TRUE)
    endif(FREEIMAGE_LIBRARY)
endif()

message(STATUS "FREEIMAGE_LIBRARY: ${FREEIMAGE_LIBRARY}")
message(STATUS "FREEIMAGE_LIBRARIES: ${FREEIMAGE_LIBRARIES}")
message(STATUS "FREEIMAGE_INCLUDE_DIRS: ${FREEIMAGE_INCLUDE_DIRS}")
