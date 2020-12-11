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

add_compile_definitions(NAPI_EXPERIMENTAL
                        NODE_ADDON_API_DISABLE_DEPRECATED)

execute_process(COMMAND node -p "require('node-addon-api').include.replace(/\"/g, '')"
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                OUTPUT_VARIABLE NAPI_INCLUDE_DIR
                OUTPUT_STRIP_TRAILING_WHITESPACE)

list(APPEND NAPI_INCLUDE_DIRS ${CMAKE_JS_INC})
list(APPEND NAPI_INCLUDE_DIRS ${NAPI_INCLUDE_DIR})

message(STATUS "CMAKE_JS_INC: ${CMAKE_JS_INC}")
message(STATUS "NAPI_INCLUDE_DIR: ${NAPI_INCLUDE_DIR}")
message(STATUS "NAPI_INCLUDE_DIRS: ${NAPI_INCLUDE_DIRS}")
