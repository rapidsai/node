// Copyright (c) 2020, NVIDIA CORPORATION.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "cuda/utilities/napi_to_cpp.hpp"
#include "macros.hpp"

#include <cuda_runtime_api.h>
#include <nv_node/utilities/args.hpp>

namespace nv {

// CUresult cuLaunchKernel(CUfunction f,
//                         unsigned int gridDimX, unsigned int gridDimY,
//                         unsigned int gridDimZ, unsigned int blockDimX,
//                         unsigned int blockDimY, unsigned int blockDimZ,
//                         unsigned int sharedMemBytes, CUstream hStream,
//                         void **kernelParams, void ** extra);
Napi::Value cuLaunchKernel(CallbackArgs const& info) {
  auto env                       = info.Env();
  CUfunction func                = info[0];
  std::vector<uint32_t> grid     = info[1];
  std::vector<uint32_t> block    = info[2];
  uint32_t sharedMem             = info[3];
  CUstream stream                = info[4];
  std::vector<napi_value> params = info[5];

  CU_TRY(env,
         CUDAAPI::cuLaunchKernel(func,
                                 grid[0],
                                 grid[1],
                                 grid[2],
                                 block[0],
                                 block[1],
                                 block[2],
                                 sharedMem,
                                 stream,
                                 (void**)params.data(),
                                 nullptr));

  return env.Undefined();
}

namespace kernel {
Napi::Object initModule(Napi::Env env, Napi::Object exports) {
  EXPORT_FUNC(env, exports, "launch", nv::cuLaunchKernel);
  return exports;
}
}  // namespace kernel
}  // namespace nv
