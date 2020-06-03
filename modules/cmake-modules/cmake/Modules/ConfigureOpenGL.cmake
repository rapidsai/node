include(FindOpenGL REQUIRED)

if(NOT OPENGL_FOUND)
    message(FATAL_ERROR "OpenGL not found")
endif()
message(STATUS "OpenGL libraries: " ${OPENGL_LIBRARIES})
message(STATUS "OpenGL includes: " ${OPENGL_INCLUDE_DIR})
