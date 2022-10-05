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

function(_statically_link_cuda_toolkit_libs target)
  if(TARGET ${target})

    get_target_property(_aliased_target ${target} ALIASED_TARGET)

    if (_aliased_target)
        _statically_link_cuda_toolkit_libs(${_aliased_target})
        return()
    endif()

    get_target_property(_link_libs ${target} INTERFACE_LINK_LIBRARIES)

    foreach(_lib IN ITEMS blas cublas cublasLt cudart cufft cufftw cupti curand
                          cusolver cusolver_lapack cusolver_metis cusparse lapack nppc
                          nppial nppicc nppicom nppidei nppif nppig nppim nppist nppisu
                          nppitc npps nvgraph)
      set(_suf "_static")
      if(_lib STREQUAL "cufft")
        set(_suf "_static_nocallback")
      endif()
      string(REPLACE "CUDA::${_lib};" "CUDA::${_lib}${_suf};" _link_libs "${_link_libs}")
      string(REPLACE "CUDA::${_lib}>" "CUDA::${_lib}${_suf}>" _link_libs "${_link_libs}")
      string(REPLACE "CUDA::${_lib}\"" "CUDA::${_lib}${_suf}\"" _link_libs "${_link_libs}")
    endforeach()

    set_target_properties(${target} PROPERTIES INTERFACE_LINK_LIBRARIES "${_link_libs}")
  endif()
endfunction()
