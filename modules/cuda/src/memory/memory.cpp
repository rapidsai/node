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

#include "node_cuda/memory.hpp"
#include "node_cuda/utilities/napi_to_cpp.hpp"

#include <nv_node/macros.hpp>

namespace nv {

namespace {

void cudaMemsetNapi(CallbackArgs const& args) {
  Span<char> target = args[0];
  int32_t value     = args[1];
  size_t count      = args[2];
  if (args.Length() == 3) {
    NODE_CUDA_TRY(cudaMemset(target.data(), value, count));
  } else {
    cudaStream_t stream = args[3];
    NODE_CUDA_TRY(cudaMemsetAsync(target.data(), value, count, stream));
  }
}

void cudaMemcpyNapi(CallbackArgs const& args) {
  Span<char> target = args[0];
  Span<char> source = args[1];
  size_t count      = args[2];
  if (args.Length() == 3) {
    NODE_CUDA_TRY(cudaMemcpy(target.data(), source.data(), count, cudaMemcpyDefault), args.Env());
  } else {
    cudaStream_t stream = args[3];
    NODE_CUDA_TRY(cudaMemcpyAsync(target.data(), source.data(), count, cudaMemcpyDefault, stream),
                  args.Env());
  }
}

// CUresult cudaMemGetInfo(size_t * free, size_t * total);
Napi::Value cudaMemGetInfoNapi(CallbackArgs const& args) {
  size_t free, total;
  NODE_CUDA_TRY(CUDARTAPI::cudaMemGetInfo(&free, &total), args.Env());
  return CPPToNapi(args)(std::vector<size_t>{free, total},
                         std::vector<std::string>{"free", "total"});
}

// CUresult cuPointerGetAttribute(void *data, CUpointer_attribute attribute,
// CUdeviceptr ptr);
Napi::Value cuPointerGetAttributeNapi(CallbackArgs const& args) {
  auto env                      = args.Env();
  CUdeviceptr dptr              = args[0];
  CUpointer_attribute attribute = args[1];

  switch (attribute) {
    case CU_POINTER_ATTRIBUTE_SYNC_MEMOPS:
    case CU_POINTER_ATTRIBUTE_IS_MANAGED: {
      bool data;
      NODE_CU_TRY(cuPointerGetAttribute(&data, attribute, dptr), env);
      return CPPToNapi(args)(data);
    }
    case CU_POINTER_ATTRIBUTE_CONTEXT: {
      CUcontext data;
      NODE_CU_TRY(cuPointerGetAttribute(&data, attribute, dptr), env);
      return CPPToNapi(args)(data);
    }
    case CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL:
    case CU_POINTER_ATTRIBUTE_MEMORY_TYPE: {
      uint32_t data;
      NODE_CU_TRY(cuPointerGetAttribute(&data, attribute, dptr), env);
      return CPPToNapi(args)(data);
    }
    case CU_POINTER_ATTRIBUTE_BUFFER_ID: {
      uint64_t data;
      NODE_CU_TRY(cuPointerGetAttribute(&data, attribute, dptr), env);
      return CPPToNapi(args)(data);
    }
    // case CU_POINTER_ATTRIBUTE_DEVICE_POINTER: {
    //   size_t size;
    //   CUdeviceptr base;
    //   CUdeviceptr data;
    //   NODE_CU_TRY(cuPointerGetAttribute(&data, attribute, dptr), env);
    //   NODE_CU_TRY(cuMemGetAddressRange(&base, &size, dptr), env);
    //   return CPPToNapi(args)(reinterpret_cast<size_t>(base));
    // }
    case CU_POINTER_ATTRIBUTE_HOST_POINTER: {
      size_t size;
      CUdeviceptr base;
      char* data{nullptr};
      NODE_CU_TRY(cuPointerGetAttribute(&data, attribute, dptr), env);
      NODE_CU_TRY(cuMemGetAddressRange(&base, &size, dptr), env);
      return CPPToNapi(args)({data, size - (dptr - base)});
    }
    // todo?
    case CU_POINTER_ATTRIBUTE_P2P_TOKENS: break;
    default: NODE_CUDA_THROW(cudaErrorNotSupported, env);
  }

  return env.Undefined();
}

}  // namespace

namespace memory {
Napi::Object initModule(Napi::Env env,
                        Napi::Object exports,
                        Napi::Object driver,
                        Napi::Object runtime) {
  nv::PinnedMemory::Init(env, exports);
  nv::DeviceMemory::Init(env, exports);
  nv::ManagedMemory::Init(env, exports);
  nv::IpcMemory::Init(env, exports);
  nv::IpcHandle::Init(env, exports);
  nv::MappedGLMemory::Init(env, exports);

  EXPORT_FUNC(env, runtime, "cudaMemset", cudaMemsetNapi);
  EXPORT_FUNC(env, runtime, "cudaMemcpy", cudaMemcpyNapi);
  EXPORT_FUNC(env, runtime, "cudaMemGetInfo", cudaMemGetInfoNapi);
  EXPORT_FUNC(env, driver, "cuPointerGetAttribute", cuPointerGetAttributeNapi);

  auto PointerAttributes = Napi::Object::New(env);
  EXPORT_ENUM(env, PointerAttributes, "context", CU_POINTER_ATTRIBUTE_CONTEXT);
  EXPORT_ENUM(env, PointerAttributes, "memory_type", CU_POINTER_ATTRIBUTE_MEMORY_TYPE);
  EXPORT_ENUM(env, PointerAttributes, "device_pointer", CU_POINTER_ATTRIBUTE_DEVICE_POINTER);
  EXPORT_ENUM(env, PointerAttributes, "host_pointer", CU_POINTER_ATTRIBUTE_HOST_POINTER);
  // EXPORT_ENUM(env, PointerAttributes, "p2p_tokens", CU_POINTER_ATTRIBUTE_P2P_TOKENS);
  EXPORT_ENUM(env, PointerAttributes, "sync_memops", CU_POINTER_ATTRIBUTE_SYNC_MEMOPS);
  EXPORT_ENUM(env, PointerAttributes, "buffer_id", CU_POINTER_ATTRIBUTE_BUFFER_ID);
  EXPORT_ENUM(env, PointerAttributes, "is_managed", CU_POINTER_ATTRIBUTE_IS_MANAGED);
  EXPORT_ENUM(env, PointerAttributes, "device_ordinal", CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL);

  EXPORT_PROP(driver, "PointerAttributes", PointerAttributes);

  return exports;
}
}  // namespace memory

}  // namespace nv
