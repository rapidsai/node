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

include(get_cpm)

set(LIBCUDACXX_GIT_REPOSITORY "https://github.com/NVIDIA/libcudacxx.git")

message(STATUS "Fetching libcudacxx from ${LIBCUDACXX_GIT_REPOSITORY}")

CPMFindPackage(NAME libcudacxx
    VERSION         1.4.0
    GIT_REPOSITORY  ${LIBCUDACXX_GIT_REPOSITORY}
    GIT_TAG         1.4.0
    GIT_SHALLOW     TRUE
    DONWLOAD_ONLY   TRUE
    OPTIONS         "LIBCXX_CONFIGURE_IDE OFF"
                    "LIBCXX_ENABLE_STATIC OFF"
                    "LIBCXX_ENABLE_SHARED OFF"
                    "LIBCXX_INCLUDE_TESTS OFF"
                    "LIBCXX_INSTALL_LIBRARY OFF"
                    "LIBCXX_INSTALL_HEADERS OFF"
                    "LIBCXX_STANDALONE_BUILD OFF"
                    "LIBCXX_DISABLE_ARCH_BY_DEFAULT ON"
                    "LIBCXX_INSTALL_SUPPORT_HEADERS OFF"
                    "LIBCXX_ENABLE_EXPERIMENTAL_LIBRARY OFF"
                    )

set(LIBCUDACXX_INCLUDE_DIR "${libcudacxx_SOURCE_DIR}/include")

message(STATUS "LIBCUDACXX_INCLUDE_DIR: ${LIBCUDACXX_INCLUDE_DIR}")
