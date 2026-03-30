#=============================================================================
# Copyright (c) 2020-2026, NVIDIA CORPORATION.
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

set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

cmake_minimum_required(VERSION 4.2.3 FATAL_ERROR)

execute_process(
  COMMAND node -p "require('@rapidsai/core').cmake_modules_path"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE rapidsai-modules
  COMMAND_ERROR_IS_FATAL ANY
)

include("${rapidsai-modules}/cmake_policies.cmake")

project(rapidsai_glfw VERSION $ENV{npm_package_version} LANGUAGES C CXX)

###################################################################################################
# - cmake modules ---------------------------------------------------------------------------------

include("${rapidsai-modules}/ConfigureCXX.cmake")
include("${rapidsai-modules}/ConfigureNapi.cmake")
include("${rapidsai-modules}/install_utils.cmake")

###################################################################################################
# - rapidsai_glfw target --------------------------------------------------------------------------

file(GLOB_RECURSE _cpp_sources "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE _cuda_sources "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu")

function(make_rapidsai_glfw_target VARIANT)

  set(options "")
  set(oneValueArgs "")
  set(multiValueArgs DEFINES)

  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(NODE_RAPIDS_CMAKE_C_FLAGS ${NODE_RAPIDS_CMAKE_C_FLAGS} ${__DEFINES})
  set(NODE_RAPIDS_CMAKE_CXX_FLAGS ${NODE_RAPIDS_CMAKE_CXX_FLAGS} ${__DEFINES})

  find_packages_and_generate_exports(
    NAME ${PROJECT_NAME}_${VARIANT}
    NPM_DEPS @rapidsai/core
    CPM_DEPS rapidsai::rapidsai_core
             glfw3_${VARIANT}
             OpenGL::EGL
             OpenGL::OpenGL
     MODULES "${rapidsai-modules}/ConfigureOpenGL.cmake"
             "${rapidsai-modules}/ConfigureOpenGLFW.cmake"
  )

  add_library(${PROJECT_NAME}_${VARIANT} SHARED ${_cpp_sources} ${_cuda_sources} ${CMAKE_JS_SRC})

  set_target_properties(${PROJECT_NAME}_${VARIANT}
    PROPERTIES PREFIX                              ""
               SUFFIX                              ".node"
               BUILD_RPATH                         "\$ORIGIN"
               INSTALL_RPATH                       "\$ORIGIN"
               CXX_STANDARD                        20
               CXX_STANDARD_REQUIRED               ON
               CUDA_STANDARD                       20
               CUDA_STANDARD_REQUIRED              ON
               NO_SYSTEM_FROM_IMPORTED             ON
               POSITION_INDEPENDENT_CODE           ON
               INTERFACE_POSITION_INDEPENDENT_CODE ON
  )

  target_compile_options(${PROJECT_NAME}_${VARIANT}
      PRIVATE "$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:C>:${NODE_RAPIDS_CMAKE_C_FLAGS}>>"
              "$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CXX>:${NODE_RAPIDS_CMAKE_CXX_FLAGS}>>"
              "$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CUDA>:${NODE_RAPIDS_CMAKE_CUDA_FLAGS}>>"
  )

  target_include_directories(${PROJECT_NAME}_${VARIANT}
       PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>"
      PRIVATE "$<BUILD_INTERFACE:${NAPI_INCLUDE_DIRS}>"
    INTERFACE "$<INSTALL_INTERFACE:../../src>"
  )

  target_link_libraries(${PROJECT_NAME}_${VARIANT}
       PUBLIC rapidsai::rapidsai_core
              glfw3_${VARIANT}
              OpenGL::EGL
              OpenGL::OpenGL
  )
endfunction()

make_rapidsai_glfw_target(x11
  DEFINES -DGLFW_EXPOSE_NATIVE_X11
)
# make_rapidsai_glfw_target(wayland
#   DEFINES -DGLFW_EXPOSE_NATIVE_WAYLAND
# )
make_rapidsai_glfw_target(eglheadless
  DEFINES -DGLFW_EXPOSE_NATIVE_EGL_EXT
          -D_GLFW_EGL_LIBRARY=libEGL_nvidia.so.0
)

generate_install_rules(
  INSTALL_TOP_LEVEL
  NAME           ${PROJECT_NAME}
  GLOBAL_TARGETS ${PROJECT_NAME}_x11
                #  ${PROJECT_NAME}_wayland
                 ${PROJECT_NAME}_eglheadless
)

# Create a symlink to compile_commands.json for the llvm-vs-code-extensions.vscode-clangd plugin
execute_process(COMMAND
    ${CMAKE_COMMAND} -E create_symlink
    ${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json
    ${CMAKE_CURRENT_SOURCE_DIR}/compile_commands.json)
