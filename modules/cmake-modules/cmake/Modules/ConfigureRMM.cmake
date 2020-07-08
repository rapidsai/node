include(FindRMM)

if(NOT RMM_FOUND OR (RMM_VERSION VERSION_LESS ${REQUIRED_RMM_VERSION}))

    message(STATUS "RMM ${REQUIRED_RMM_VERSION} not found, building from source")

    set(RMM_ROOT "${CMAKE_BINARY_DIR}/rmm")

    set(RMM_GIT_BRANCH_NAME "branch-${REQUIRED_RMM_VERSION}")

    set(RMM_CMAKE_ARGS " -DBUILD_TESTS=OFF"
                       " -DBUILD_BENCHMARKS=OFF")

    configure_file("${CMAKE_CURRENT_LIST_DIR}/../Templates/RMM.CMakeLists.txt.cmake"
                   "${RMM_ROOT}/CMakeLists.txt")

    execute_process(
        COMMAND ${CMAKE_COMMAND} -Wno-dev -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE RMM_CONFIG
        WORKING_DIRECTORY ${RMM_ROOT})

    if(RMM_CONFIG)
        message(FATAL_ERROR "Configuring RMM failed: " ${RMM_CONFIG})
    endif(RMM_CONFIG)

    execute_process(
        COMMAND ${CMAKE_COMMAND} --build .. -- ${PARALLEL_BUILD}
        RESULT_VARIABLE RMM_BUILD
        WORKING_DIRECTORY ${RMM_ROOT}/build)

    if(RMM_BUILD)
        message(FATAL_ERROR "Building RMM failed: " ${RMM_BUILD})
    endif(RMM_BUILD)

    set(RMM_LIBRARIES "${CMAKE_BINARY_DIR}/lib")
    set(RMM_INCLUDE_DIRS "${CMAKE_BINARY_DIR}/include")

    find_library(RMM_LIBRARY rmm NO_DEFAULT_PATH HINTS "${RMM_LIBRARIES}")

    if(RMM_LIBRARY)
        set(RMM_FOUND TRUE)
    endif(RMM_LIBRARY)
endif()

message(STATUS "RMM_LIBRARY: ${RMM_LIBRARY}")
message(STATUS "RMM_LIBRARIES: ${RMM_LIBRARIES}")
message(STATUS "RMM_INCLUDE_DIRS: ${RMM_INCLUDE_DIRS}")
