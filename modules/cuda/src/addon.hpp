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

#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <napi.h>

namespace nv {

namespace device {
Napi::Object initModule(Napi::Env env, Napi::Object exports);
}  // namespace device

namespace gl {
Napi::Object initModule(Napi::Env env, Napi::Object exports);
}  // namespace gl

namespace ipc {
Napi::Object initModule(Napi::Env env, Napi::Object exports);
}  // namespace ipc

namespace kernel {
Napi::Object initModule(Napi::Env env, Napi::Object exports);
}  // namespace kernel

namespace math {
Napi::Object initModule(Napi::Env env, Napi::Object exports);
}  // namespace math

namespace mem {
Napi::Object initModule(Napi::Env env, Napi::Object exports);
}  // namespace mem

namespace program {
Napi::Object initModule(Napi::Env env, Napi::Object exports);
}  // namespace program

namespace stream {
Napi::Object initModule(Napi::Env env, Napi::Object exports);
}  // namespace stream

namespace texture {
Napi::Object initModule(Napi::Env env, Napi::Object exports);
}  // namespace texture

namespace detail {
void freeHostPtr(Napi::Env const& env, void* ptr);
}  // namespace detail

}  // namespace nv
