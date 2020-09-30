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

#include "buffer.hpp"
#include "macros.hpp"
#include "task.hpp"
#include "utilities/cpp_to_napi.hpp"
#include "utilities/napi_to_cpp.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <napi.h>
#include <nv_node/utilities/args.hpp>
#include <nv_node/utilities/span.hpp>

namespace nv {

namespace detail {
void freeHostPtr(Napi::Env const& env, void* ptr) {
  size_t size;
  CUdeviceptr base;
  CUdeviceptr dptr;
  CU_TRY_VOID(env, cuMemHostGetDevicePointer(&dptr, ptr, 0));
  CU_TRY_VOID(env, cuMemGetAddressRange(&base, &size, dptr));
  if (dptr != 0 && size > 0 && cudaFreeHost(ptr) == cudaSuccess) {
    Napi::MemoryManagement::AdjustExternalMemory(env, -size);
  }
}
}  // namespace detail

// cudaError_t cudaMalloc(void **devPtr, size_t size);
// Napi::Value cudaMalloc(CallbackArgs const& info) {
//   auto env = info.Env();
//   void* data{nullptr};
//   size_t size = info[0];
//   if (size > 0) {
//     CUDA_TRY(env, CUDARTAPI::cudaMalloc(&data, size));
//     Napi::MemoryManagement::AdjustExternalMemory(env, size);
//   }
//   return CUDABuffer::New(data, size);
// }

Napi::Value cudaMalloc(CallbackArgs const& info) {
  auto env = info.Env();
  void* data{nullptr};
  size_t size = info[0];
  if (size > 0) {
    CUDA_TRY(env, CUDARTAPI::cudaMalloc(&data, size));
    Napi::MemoryManagement::AdjustExternalMemory(env, size);
  }
  return CUDABuffer::New(data, size);
}

// cudaError_t cudaFree(void *devPtr);
Napi::Value cudaFree(CallbackArgs const& info) {
  auto env = info.Env();
  auto obj = info[0].val.As<Napi::Object>();
  CUDABuffer::Unwrap(obj)->Finalize(env);
  return env.Undefined();
}

// cudaError_t cudaMallocHost(void **ptr, size_t size);
Napi::Value cudaMallocHost(CallbackArgs const& info) {
  auto env = info.Env();
  void* data{nullptr};
  size_t size = info[0];
  if (size > 0) {
    CUDA_TRY(env, CUDARTAPI::cudaMallocHost(&data, size));
    Napi::MemoryManagement::AdjustExternalMemory(env, size);
  }
  auto ary = CPPToNapi(info)(Span<unsigned char>(data, size), detail::freeHostPtr);
  return ary.As<Napi::Uint8Array>().ArrayBuffer();
}

// cudaError_t cudaFreeHost(void *ptr);
Napi::Value cudaFreeHost(CallbackArgs const& info) {
  auto env              = info.Env();
  Napi::ArrayBuffer buf = info[0];
  if (buf.Data() != nullptr) { detail::freeHostPtr(env, buf.Data()); }
  return env.Undefined();
}

// cudaError_t cudaHostRegister(void *ptr, size_t size, unsigned int flags);
Napi::Value cudaHostRegister(CallbackArgs const& info) {
  auto env              = info.Env();
  Napi::ArrayBuffer buf = info[0];
  size_t size           = buf.ByteLength();
  uint32_t flags        = info[1];
  if (buf.Data() != nullptr) {
    CUDA_TRY(env, CUDARTAPI::cudaHostRegister(buf.Data(), size, flags));
  }
  return env.Undefined();
}

// cudaError_t cudaHostUnregister(void *ptr);
Napi::Value cudaHostUnregister(CallbackArgs const& info) {
  auto env              = info.Env();
  Napi::ArrayBuffer buf = info[0];
  if (buf.Data() != nullptr) { CUDA_TRY(env, CUDARTAPI::cudaHostUnregister(buf.Data())); }
  return env.Undefined();
}

// cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum
// cudaMemcpyKind kind);
Napi::Value cudaMemcpy(CallbackArgs const& info) {
  auto env          = info.Env();
  uint8_t* dst_data = info[0];
  size_t dst_offset = info[1];
  uint8_t* src_data = info[2];
  size_t src_offset = info[3];
  size_t size       = info[4];
  if (dst_data != nullptr && src_data != nullptr && size > 0) {
    CUDA_TRY(
      env,
      CUDARTAPI::cudaMemcpy(dst_data + dst_offset, src_data + src_offset, size, cudaMemcpyDefault));
  }
  return env.Undefined();
}

Napi::Value cudaMemcpy2D(CallbackArgs const& info) {
  auto env         = info.Env();
  uint8_t* dst_ary = info[0];
  size_t dst_pitch = info[1];
  uint8_t* src_ary = info[2];
  size_t src_pitch = info[3];
  size_t width     = info[4];
  size_t height    = info[5];
  CUDA_TRY(env,
           CUDARTAPI::cudaMemcpy2D(
             dst_ary, dst_pitch, src_ary, src_pitch, width, height, cudaMemcpyDefault));
  return env.Undefined();
}

Napi::Value cudaMemcpy2DFromArray(CallbackArgs const& info) {
  auto env            = info.Env();
  void* dst_ary       = info[0];
  size_t dst_pitch    = info[1];
  cudaArray_t src_ary = info[2];
  size_t x            = info[3];
  size_t y            = info[4];
  size_t width        = info[5];
  size_t height       = info[6];
  CUDA_TRY(env,
           CUDARTAPI::cudaMemcpy2DFromArray(
             dst_ary, dst_pitch, src_ary, x, y, width, height, cudaMemcpyDefault));
  return env.Undefined();
}

// cudaError_t cudaMemset(void *devPtr, int value, size_t count);
Napi::Value cudaMemset(CallbackArgs const& info) {
  auto env      = info.Env();
  uint8_t* data = info[0];
  size_t offset = info[1];
  int32_t value = info[2];
  size_t count  = info[3];
  if (data != nullptr) { CUDA_TRY(env, CUDARTAPI::cudaMemset(data + offset, value, count)); }
  return env.Undefined();
}

// cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, enum
// cudaMemcpyKind kind, cudaStream_t stream);
Napi::Value cudaMemcpyAsync(CallbackArgs const& info) {
  auto env            = info.Env();
  uint8_t* dst_data   = info[0];
  size_t dst_offset   = info[1];
  uint8_t* src_data   = info[2];
  size_t src_offset   = info[3];
  size_t size         = info[4];
  cudaStream_t stream = info[5];
  auto task           = new nv::Task(env);
  if (task->DelayResolve(dst_data != nullptr && src_data != nullptr && size > 0)) {
    CUDA_TRY_ASYNC(
      task,
      CUDARTAPI::cudaMemcpyAsync(
        dst_data + dst_offset, src_data + src_offset, size, cudaMemcpyDefault, stream));
    CUDA_TRY_ASYNC(task, CUDARTAPI::cudaLaunchHostFunc(stream, nv::Task::Notify, task));
  }
  return task->Promise();
}

// cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count,
// cudaStream_t stream);
Napi::Value cudaMemsetAsync(CallbackArgs const& info) {
  auto env            = info.Env();
  uint8_t* data       = info[0];
  size_t offset       = info[1];
  int32_t value       = info[2];
  size_t count        = info[3];
  cudaStream_t stream = info[4];
  auto task           = new nv::Task(env);
  if (task->DelayResolve(data != nullptr && count > 0)) {
    CUDA_TRY_ASYNC(task, CUDARTAPI::cudaMemsetAsync(data + offset, value, count, stream));
    CUDA_TRY_ASYNC(task, CUDARTAPI::cudaLaunchHostFunc(stream, nv::Task::Notify, task));
  }
  return task->Promise();
}

// CUresult cudaMemGetInfo(size_t * free, size_t * total);
Napi::Value cudaMemGetInfo(CallbackArgs const& info) {
  auto env = info.Env();
  size_t free, total;
  CUDA_TRY(env, CUDARTAPI::cudaMemGetInfo(&free, &total));
  return CPPToNapi(info)(std::vector<size_t>{free, total},
                         std::vector<std::string>{"free", "total"});
}

// CUresult cuPointerGetAttribute(void *data, CUpointer_attribute attribute,
// CUdeviceptr ptr);
Napi::Value cuPointerGetAttribute(CallbackArgs const& info) {
  auto env                      = info.Env();
  CUdeviceptr dptr              = info[0];
  CUpointer_attribute attribute = info[1];

  switch (attribute) {
    case CU_POINTER_ATTRIBUTE_SYNC_MEMOPS:
    case CU_POINTER_ATTRIBUTE_IS_MANAGED: {
      bool data;
      CU_TRY(env, CUDAAPI::cuPointerGetAttribute(&data, attribute, dptr));
      return CPPToNapi(info)(data);
    }
    case CU_POINTER_ATTRIBUTE_CONTEXT: {
      CUcontext data;
      CU_TRY(env, CUDAAPI::cuPointerGetAttribute(&data, attribute, dptr));
      return CPPToNapi(info)(data);
    }
    case CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL:
    case CU_POINTER_ATTRIBUTE_MEMORY_TYPE: {
      uint32_t data;
      CU_TRY(env, CUDAAPI::cuPointerGetAttribute(&data, attribute, dptr));
      return CPPToNapi(info)(data);
    }
    case CU_POINTER_ATTRIBUTE_BUFFER_ID: {
      uint64_t data;
      CU_TRY(env, CUDAAPI::cuPointerGetAttribute(&data, attribute, dptr));
      return CPPToNapi(info)(data);
    }
    // case CU_POINTER_ATTRIBUTE_DEVICE_POINTER: {
    //   size_t size;
    //   CUdeviceptr base;
    //   CUdeviceptr data;
    //   CU_TRY(env, CUDAAPI::cuPointerGetAttribute(&data, attribute, dptr));
    //   CU_TRY(env, CUDAAPI::cuMemGetAddressRange(&base, &size, dptr));
    //   return CUDABuffer::New(reinterpret_cast<void*>(data), size);
    // }
    case CU_POINTER_ATTRIBUTE_HOST_POINTER: {
      size_t size;
      CUdeviceptr base;
      char* data{nullptr};
      CU_TRY(env, CUDAAPI::cuPointerGetAttribute(&data, attribute, dptr));
      CU_TRY(env, CUDAAPI::cuMemGetAddressRange(&base, &size, dptr));
      return CPPToNapi(info)(data, size - (dptr - base));
    }
    // todo?
    case CU_POINTER_ATTRIBUTE_P2P_TOKENS: break;
    default: CUDA_THROW(env, cudaErrorNotSupported);
  }

  return env.Undefined();
}

namespace mem {

Napi::Object initModule(Napi::Env env, Napi::Object exports) {
  nv::CUDABuffer::Init(env, exports);

  EXPORT_FUNC(env, exports, "alloc", nv::cudaMalloc);
  EXPORT_FUNC(env, exports, "free", nv::cudaFree);
  EXPORT_FUNC(env, exports, "allocHost", nv::cudaMallocHost);
  EXPORT_FUNC(env, exports, "freeHost", nv::cudaFreeHost);
  EXPORT_FUNC(env, exports, "hostRegister", nv::cudaHostRegister);
  EXPORT_FUNC(env, exports, "hostUnregister", nv::cudaHostUnregister);
  EXPORT_FUNC(env, exports, "cpy", nv::cudaMemcpy);
  EXPORT_FUNC(env, exports, "cpy2D", nv::cudaMemcpy2D);
  EXPORT_FUNC(env, exports, "cpy2DFromArray", nv::cudaMemcpy2DFromArray);
  EXPORT_FUNC(env, exports, "set", nv::cudaMemset);
  EXPORT_FUNC(env, exports, "cpyAsync", nv::cudaMemcpyAsync);
  EXPORT_FUNC(env, exports, "setAsync", nv::cudaMemsetAsync);
  EXPORT_FUNC(env, exports, "getInfo", nv::cudaMemGetInfo);
  EXPORT_FUNC(env, exports, "getPointerAttribute", nv::cuPointerGetAttribute);

  auto cudaMemoryTypeFlags = Napi::Object::New(env);
  EXPORT_ENUM(env, cudaMemoryTypeFlags, "unregistered", cudaMemoryTypeUnregistered);
  EXPORT_ENUM(env, cudaMemoryTypeFlags, "host", cudaMemoryTypeHost);
  EXPORT_ENUM(env, cudaMemoryTypeFlags, "device", cudaMemoryTypeDevice);
  EXPORT_ENUM(env, cudaMemoryTypeFlags, "managed", cudaMemoryTypeManaged);

  auto cudaHostAllocFlags = Napi::Object::New(env);
  EXPORT_ENUM(env, cudaHostAllocFlags, "default", cudaHostAllocDefault);
  EXPORT_ENUM(env, cudaHostAllocFlags, "portable", cudaHostAllocPortable);
  EXPORT_ENUM(env, cudaHostAllocFlags, "mapped", cudaHostAllocMapped);
  EXPORT_ENUM(env, cudaHostAllocFlags, "writeCombined", cudaHostAllocWriteCombined);

  auto cudaHostRegisterFlags = Napi::Object::New(env);
  EXPORT_ENUM(env, cudaHostRegisterFlags, "default", cudaHostRegisterDefault);
  EXPORT_ENUM(env, cudaHostRegisterFlags, "portable", cudaHostRegisterPortable);
  EXPORT_ENUM(env, cudaHostRegisterFlags, "mapped", cudaHostRegisterMapped);
  EXPORT_ENUM(env, cudaHostRegisterFlags, "ioMemory", cudaHostRegisterIoMemory);

  auto cuPointerAttributeFlags = Napi::Object::New(env);
  EXPORT_ENUM(env, cuPointerAttributeFlags, "context", CU_POINTER_ATTRIBUTE_CONTEXT);
  EXPORT_ENUM(env, cuPointerAttributeFlags, "memory_type", CU_POINTER_ATTRIBUTE_MEMORY_TYPE);
  EXPORT_ENUM(env, cuPointerAttributeFlags, "device_pointer", CU_POINTER_ATTRIBUTE_DEVICE_POINTER);
  EXPORT_ENUM(env, cuPointerAttributeFlags, "host_pointer", CU_POINTER_ATTRIBUTE_HOST_POINTER);
  // EXPORT_ENUM(env, cuPointerAttributeFlags, "p2p_tokens",
  // CU_POINTER_ATTRIBUTE_P2P_TOKENS);
  EXPORT_ENUM(env, cuPointerAttributeFlags, "sync_memops", CU_POINTER_ATTRIBUTE_SYNC_MEMOPS);
  EXPORT_ENUM(env, cuPointerAttributeFlags, "buffer_id", CU_POINTER_ATTRIBUTE_BUFFER_ID);
  EXPORT_ENUM(env, cuPointerAttributeFlags, "is_managed", CU_POINTER_ATTRIBUTE_IS_MANAGED);
  EXPORT_ENUM(env, cuPointerAttributeFlags, "device_ordinal", CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL);

  EXPORT_PROP(exports, "memoryTypes", cudaMemoryTypeFlags);
  EXPORT_PROP(exports, "hostAllocFlags", cudaHostAllocFlags);
  EXPORT_PROP(exports, "hostRegisterFlags", cudaHostRegisterFlags);
  EXPORT_PROP(exports, "pointerAttributes", cuPointerAttributeFlags);

  return exports;
}
}  // namespace mem
}  // namespace nv
