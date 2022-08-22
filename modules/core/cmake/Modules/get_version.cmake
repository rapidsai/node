#=============================================================================
# Copyright (c) 2022, NVIDIA CORPORATION.
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
include_guard(GLOBAL)

function(_get_rapidsai_module_version pkg out_var_)
  set(ver_ "22.08.00")
  if(DEFINED ${pkg}_VERSION)
    set(ver_ "${${pkg}_VERSION}")
  elseif(DEFINED RAPIDS_VERSION)
    set(ver_ "${RAPIDS_VERSION}")
  elseif(DEFINED ENV{RAPIDS_VERSION})
    set(ver_ "$ENV{RAPIDS_VERSION}")
    set(RAPIDS_VERSION "${ver_}" PARENT_SCOPE)
  endif()
  set(${out_var_} "${ver_}" PARENT_SCOPE)
endfunction()
