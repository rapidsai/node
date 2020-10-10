// Copyright(c) 2020, NVIDIA CORPORATION.
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

#include "cuda_memory_resource.hpp"
#include "macros.hpp"
#include "napi_to_cpp.hpp"

#include <node_cuda/utilities/napi_to_cpp.hpp>
#include <nv_node/utilities/args.hpp>
#include <nv_node/utilities/cpp_to_napi.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

namespace nv {

//
// Public API
//

Napi::FunctionReference CudaMemoryResource::constructor;

Napi::Object CudaMemoryResource::Init(Napi::Env env, Napi::Object exports) {
  const Napi::Function ctor = DefineClass(
    env,
    "CudaMemoryResource",
    {
      InstanceAccessor(
        "supportsGetMemInfo", &CudaMemoryResource::supportsGetMemInfo, nullptr, napi_enumerable),
      InstanceAccessor(
        "supportsStreams", &CudaMemoryResource::supportsStreams, nullptr, napi_enumerable),
      InstanceMethod("allocate", &CudaMemoryResource::allocate),
      InstanceMethod("deallocate", &CudaMemoryResource::deallocate),
      InstanceMethod("getMemInfo", &CudaMemoryResource::getMemInfo),
      InstanceMethod("isEqual", &CudaMemoryResource::isEqual),
    });
  CudaMemoryResource::constructor = Napi::Persistent(ctor);
  CudaMemoryResource::constructor.SuppressDestruct();
  exports.Set("CudaMemoryResource", ctor);
  return exports;
}

CudaMemoryResource::CudaMemoryResource(Napi::CallbackInfo const& info)
  : Napi::ObjectWrap<CudaMemoryResource>(info) {
  Initialize();
}

Napi::Value CudaMemoryResource::New() {
  const auto inst = CudaMemoryResource::constructor.New({});
  CudaMemoryResource::Unwrap(inst)->Initialize();
  return inst;
}

void CudaMemoryResource::Initialize() { resource_.reset(new rmm::mr::cuda_memory_resource()); }

void CudaMemoryResource::Finalize(Napi::Env env) {
  if (resource_ != nullptr) { this->resource_ = nullptr; }
  resource_ = nullptr;
}

//
// Private API
//

Napi::Value CudaMemoryResource::allocate(Napi::CallbackInfo const& info) {
  const CallbackArgs args{info};
  const size_t bytes    = args[0];
  void* const allocated = [&] {
    if (args.Length() > 1 && info[1].IsNumber()) {
      const cudaStream_t stream = args[1];
      return Resource()->allocate(bytes, stream);
    } else {
      return Resource()->allocate(bytes);
    }
  }();
  return CPPToNapi(info)(allocated);
}

Napi::Value CudaMemoryResource::deallocate(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  void* const ptr    = args[0];
  const size_t bytes = args[1];
  if (args.Length() > 2 && info[2].IsNumber()) {
    const cudaStream_t stream = args[2];
    Resource()->deallocate(ptr, bytes, stream);
  } else {
    Resource()->deallocate(ptr, bytes);
  }
  return info.Env().Undefined();
}

Napi::Value CudaMemoryResource::getMemInfo(Napi::CallbackInfo const& info) {
  const CallbackArgs args{info};
  const cudaStream_t stream = args[0];
  return CPPToNapi(info)(Resource()->get_mem_info(stream));
}

Napi::Value CudaMemoryResource::isEqual(Napi::CallbackInfo const& info) {
  const CallbackArgs args{info};
  CudaMemoryResource const& other = args[0];
  return CPPToNapi(info)(Resource()->is_equal(*other.Resource()));
}

Napi::Value CudaMemoryResource::supportsStreams(Napi::CallbackInfo const& info) {
  return CPPToNapi(info)(Resource()->supports_streams());
}

Napi::Value CudaMemoryResource::supportsGetMemInfo(Napi::CallbackInfo const& info) {
  return CPPToNapi(info)(Resource()->supports_get_mem_info());
}

}  // namespace nv
