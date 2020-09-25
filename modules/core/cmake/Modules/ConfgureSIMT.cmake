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

# ConfigureSimt.cmake
include(get_cpm)
CPMAddPackage(NAME libcudacxx
    VERSION        0.0.1
    GIT_REPOSITORY https://github.com/rapidsai/thirdparty-freestanding.git
    GIT_TAG        cudf
    GIT_SHALLOW    FALSE
    DOWNLOAD_ONLY
)
message(STATUS "libcudacxx source dir: " ${libcudacxx_SOURCE_DIR})
set(LIBCUDACXX_INCLUDE_DIR "${libcudacxx_SOURCE_DIR}/include")