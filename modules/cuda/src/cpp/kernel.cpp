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

#include <cuda_runtime.h>

#include <node_cuda/casting.hpp>
#include <node_cuda/macros.hpp>

namespace node_cuda {

// CUresult cuLaunchKernel(CUfunction f,
//                         unsigned int gridDimX, unsigned int gridDimY,
//                         unsigned int gridDimZ, unsigned int blockDimX,
//                         unsigned int blockDimY, unsigned int blockDimZ,
//                         unsigned int sharedMemBytes, CUstream hStream,
//                         void **kernelParams, void ** extra);
Napi::Value cuLaunchKernel(Napi::CallbackInfo const& info) {
  auto env                       = info.Env();
  CUfunction func                = FromJS(info[0]);
  std::vector<uint32_t> grid     = FromJS(info[1]);
  std::vector<uint32_t> block    = FromJS(info[2]);
  uint32_t sharedMem             = FromJS(info[3]);
  CUstream stream                = FromJS(info[4]);
  std::vector<napi_value> params = FromJS(info[5]);

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
  EXPORT_FUNC(env, exports, "launch", node_cuda::cuLaunchKernel);
  return exports;
}
}  // namespace kernel
}  // namespace node_cuda
