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

#include <node_cuda/buffer.hpp>
#include <node_cuda/casting.hpp>
#include <node_cuda/macros.hpp>
#include <node_cuda/task.hpp>

#include <cuda.h>
#include <cuda_runtime.h>
#include <napi.h>

namespace node_cuda {

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
Napi::Value cudaMalloc(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  void* data{nullptr};
  size_t size = FromJS(info[0]);
  if (size > 0) {
    CUDA_TRY(env, CUDARTAPI::cudaMalloc(&data, size));
    Napi::MemoryManagement::AdjustExternalMemory(env, size);
  }
  return CUDABuffer::New(data, size);
}

// cudaError_t cudaFree(void *devPtr);
Napi::Value cudaFree(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  auto obj = info[0].As<Napi::Object>();
  CUDABuffer::Unwrap(obj)->Finalize(env);
  return env.Undefined();
}

// cudaError_t cudaMallocHost(void **ptr, size_t size);
Napi::Value cudaMallocHost(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  void* data{nullptr};
  size_t size = FromJS(info[0]);
  if (size > 0) {
    CUDA_TRY(env, CUDARTAPI::cudaMallocHost(&data, size));
    Napi::MemoryManagement::AdjustExternalMemory(env, size);
  }
  return ToNapi(env)(data, size, detail::freeHostPtr);
}

// cudaError_t cudaFreeHost(void *ptr);
Napi::Value cudaFreeHost(Napi::CallbackInfo const& info) {
  auto env              = info.Env();
  Napi::ArrayBuffer buf = FromJS(info[0]);
  if (buf.Data() != nullptr) { detail::freeHostPtr(env, buf.Data()); }
  return env.Undefined();
}

// cudaError_t cudaHostRegister(void *ptr, size_t size, unsigned int flags);
Napi::Value cudaHostRegister(Napi::CallbackInfo const& info) {
  auto env              = info.Env();
  Napi::ArrayBuffer buf = FromJS(info[0]);
  size_t size           = buf.ByteLength();
  uint32_t flags        = FromJS(info[1]);
  if (buf.Data() != nullptr) {
    CUDA_TRY(env, CUDARTAPI::cudaHostRegister(buf.Data(), size, flags));
  }
  return env.Undefined();
}

// cudaError_t cudaHostUnregister(void *ptr);
Napi::Value cudaHostUnregister(Napi::CallbackInfo const& info) {
  auto env              = info.Env();
  Napi::ArrayBuffer buf = FromJS(info[0]);
  if (buf.Data() != nullptr) { CUDA_TRY(env, CUDARTAPI::cudaHostUnregister(buf.Data())); }
  return env.Undefined();
}

// cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum
// cudaMemcpyKind kind);
Napi::Value cudaMemcpy(Napi::CallbackInfo const& info) {
  auto env          = info.Env();
  uint8_t* dst_data = FromJS(info[0]);
  size_t dst_offset = FromJS(info[1]);
  uint8_t* src_data = FromJS(info[2]);
  size_t src_offset = FromJS(info[3]);
  size_t size       = FromJS(info[4]);
  if (dst_data != nullptr && src_data != nullptr && size > 0) {
    CUDA_TRY(
      env,
      CUDARTAPI::cudaMemcpy(dst_data + dst_offset, src_data + src_offset, size, cudaMemcpyDefault));
  }
  return env.Undefined();
}

Napi::Value cudaMemcpy2D(Napi::CallbackInfo const& info) {
  auto env         = info.Env();
  uint8_t* dst_ary = FromJS(info[0]);
  size_t dst_pitch = FromJS(info[1]);
  uint8_t* src_ary = FromJS(info[2]);
  size_t src_pitch = FromJS(info[3]);
  size_t width     = FromJS(info[4]);
  size_t height    = FromJS(info[5]);
  CUDA_TRY(env,
           CUDARTAPI::cudaMemcpy2D(
             dst_ary, dst_pitch, src_ary, src_pitch, width, height, cudaMemcpyDefault));
  return env.Undefined();
}

Napi::Value cudaMemcpy2DFromArray(Napi::CallbackInfo const& info) {
  auto env            = info.Env();
  void* dst_ary       = FromJS(info[0]);
  size_t dst_pitch    = FromJS(info[1]);
  cudaArray_t src_ary = FromJS(info[2]);
  size_t x            = FromJS(info[3]);
  size_t y            = FromJS(info[4]);
  size_t width        = FromJS(info[5]);
  size_t height       = FromJS(info[6]);
  CUDA_TRY(env,
           CUDARTAPI::cudaMemcpy2DFromArray(
             dst_ary, dst_pitch, src_ary, x, y, width, height, cudaMemcpyDefault));
  return env.Undefined();
}

// cudaError_t cudaMemset(void *devPtr, int value, size_t count);
Napi::Value cudaMemset(Napi::CallbackInfo const& info) {
  auto env      = info.Env();
  uint8_t* data = FromJS(info[0]);
  size_t offset = FromJS(info[1]);
  int32_t value = FromJS(info[2]);
  size_t count  = FromJS(info[3]);
  if (data != nullptr) { CUDA_TRY(env, CUDARTAPI::cudaMemset(data + offset, value, count)); }
  return env.Undefined();
}

// cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, enum
// cudaMemcpyKind kind, cudaStream_t stream);
Napi::Value cudaMemcpyAsync(Napi::CallbackInfo const& info) {
  auto env            = info.Env();
  uint8_t* dst_data   = FromJS(info[0]);
  size_t dst_offset   = FromJS(info[1]);
  uint8_t* src_data   = FromJS(info[2]);
  size_t src_offset   = FromJS(info[3]);
  size_t size         = FromJS(info[4]);
  cudaStream_t stream = FromJS(info[5]);
  auto task           = new node_cuda::Task(env);
  if (task->DelayResolve(dst_data != nullptr && src_data != nullptr && size > 0)) {
    CUDA_TRY_ASYNC(
      task,
      CUDARTAPI::cudaMemcpyAsync(
        dst_data + dst_offset, src_data + src_offset, size, cudaMemcpyDefault, stream));
    CUDA_TRY_ASYNC(task, CUDARTAPI::cudaLaunchHostFunc(stream, node_cuda::Task::Notify, task));
  }
  return task->Promise();
}

// cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count,
// cudaStream_t stream);
Napi::Value cudaMemsetAsync(Napi::CallbackInfo const& info) {
  auto env            = info.Env();
  uint8_t* data       = FromJS(info[0]);
  size_t offset       = FromJS(info[1]);
  int32_t value       = FromJS(info[2]);
  size_t count        = FromJS(info[3]);
  cudaStream_t stream = FromJS(info[4]);
  auto task           = new node_cuda::Task(env);
  if (task->DelayResolve(data != nullptr && count > 0)) {
    CUDA_TRY_ASYNC(task, CUDARTAPI::cudaMemsetAsync(data + offset, value, count, stream));
    CUDA_TRY_ASYNC(task, CUDARTAPI::cudaLaunchHostFunc(stream, node_cuda::Task::Notify, task));
  }
  return task->Promise();
}

// CUresult cudaMemGetInfo(size_t * free, size_t * total);
Napi::Value cudaMemGetInfo(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  size_t free, total;
  CUDA_TRY(env, CUDARTAPI::cudaMemGetInfo(&free, &total));
  return ToNapi(env)(std::vector<size_t>{free, total}, std::vector<std::string>{"free", "total"});
}

// CUresult cuPointerGetAttribute(void *data, CUpointer_attribute attribute,
// CUdeviceptr ptr);
Napi::Value cuPointerGetAttribute(Napi::CallbackInfo const& info) {
  auto env                      = info.Env();
  CUdeviceptr dptr              = FromJS(info[0]);
  CUpointer_attribute attribute = FromJS(info[1]);

  switch (attribute) {
    case CU_POINTER_ATTRIBUTE_SYNC_MEMOPS:
    case CU_POINTER_ATTRIBUTE_IS_MANAGED: {
      bool data;
      CU_TRY(env, CUDAAPI::cuPointerGetAttribute(&data, attribute, dptr));
      return ToNapi(env)(data);
    }
    case CU_POINTER_ATTRIBUTE_CONTEXT: {
      CUcontext data;
      CU_TRY(env, CUDAAPI::cuPointerGetAttribute(&data, attribute, dptr));
      return ToNapi(env)(data);
    }
    case CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL:
    case CU_POINTER_ATTRIBUTE_MEMORY_TYPE: {
      uint32_t data;
      CU_TRY(env, CUDAAPI::cuPointerGetAttribute(&data, attribute, dptr));
      return ToNapi(env)(data);
    }
    case CU_POINTER_ATTRIBUTE_BUFFER_ID: {
      uint64_t data;
      CU_TRY(env, CUDAAPI::cuPointerGetAttribute(&data, attribute, dptr));
      return ToNapi(env)(data);
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
      void* data{nullptr};
      CU_TRY(env, CUDAAPI::cuPointerGetAttribute(&data, attribute, dptr));
      CU_TRY(env, CUDAAPI::cuMemGetAddressRange(&base, &size, dptr));
      return ToNapi(env)(data, size - (dptr - base));
    }
    // todo?
    case CU_POINTER_ATTRIBUTE_P2P_TOKENS: break;
    default: CUDA_THROW(env, cudaErrorNotSupported);
  }

  return env.Undefined();
}

namespace mem {

Napi::Object initModule(Napi::Env env, Napi::Object exports) {
  node_cuda::CUDABuffer::Init(env, exports);

  EXPORT_FUNC(env, exports, "alloc", node_cuda::cudaMalloc);
  EXPORT_FUNC(env, exports, "free", node_cuda::cudaFree);
  EXPORT_FUNC(env, exports, "allocHost", node_cuda::cudaMallocHost);
  EXPORT_FUNC(env, exports, "freeHost", node_cuda::cudaFreeHost);
  EXPORT_FUNC(env, exports, "hostRegister", node_cuda::cudaHostRegister);
  EXPORT_FUNC(env, exports, "hostUnregister", node_cuda::cudaHostUnregister);
  EXPORT_FUNC(env, exports, "cpy", node_cuda::cudaMemcpy);
  EXPORT_FUNC(env, exports, "cpy2D", node_cuda::cudaMemcpy2D);
  EXPORT_FUNC(env, exports, "cpy2DFromArray", node_cuda::cudaMemcpy2DFromArray);
  EXPORT_FUNC(env, exports, "set", node_cuda::cudaMemset);
  EXPORT_FUNC(env, exports, "cpyAsync", node_cuda::cudaMemcpyAsync);
  EXPORT_FUNC(env, exports, "setAsync", node_cuda::cudaMemsetAsync);
  EXPORT_FUNC(env, exports, "getInfo", node_cuda::cudaMemGetInfo);
  EXPORT_FUNC(env, exports, "getPointerAttribute", node_cuda::cuPointerGetAttribute);

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
}  // namespace node_cuda
