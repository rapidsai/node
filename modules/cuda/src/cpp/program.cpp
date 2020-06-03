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

// nvrtcCreateProgram(nvrtcProgram *prog,
//                    const char *src,
//                    const char *name,
//                    int numHeaders,
//                    const char * const *headers,
//                    const char * const *includeNames)
Napi::Value createProgram(Napi::CallbackInfo const& info) {
  auto env                          = info.Env();
  std::string src                   = FromJS(info[0]);
  std::string name                  = FromJS(info[1]);
  std::vector<std::string> headers  = FromJS(info[2]);
  std::vector<std::string> includes = FromJS(info[3]);

  std::vector<const char*> cHeaders(headers.size());
  std::vector<const char*> cIncludes(includes.size());

  auto get_cstr = [](const std::string& str) { return str.c_str(); };
  std::transform(headers.begin(), headers.end(), cHeaders.begin(), get_cstr);
  std::transform(includes.begin(), includes.end(), cIncludes.begin(), get_cstr);

  nvrtcProgram prog;

  NVRTC_TRY(env,
            nvrtcCreateProgram(
              &prog, src.c_str(), name.c_str(), headers.size(), cHeaders.data(), cIncludes.data()));

  auto free_str = [](const char* str) { delete str; };
  std::for_each(cHeaders.begin(), cHeaders.end(), free_str);
  std::for_each(cIncludes.begin(), cIncludes.end(), free_str);

  return ToNapi(env)(prog);
}

namespace program {
Napi::Object initModule(Napi::Env env, Napi::Object exports) {
  EXPORT_FUNC(env, exports, "create", node_cuda::createProgram);
  return exports;
}
}  // namespace program
}  // namespace node_cuda
