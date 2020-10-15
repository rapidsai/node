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

#include "node_cuda/array.hpp"
#include "node_cuda/memory.hpp"
#include "node_cuda/utilities/cpp_to_napi.hpp"
#include "node_cuda/utilities/napi_to_cpp.hpp"

#include <GL/gl.h>
#include <cuda.h>
#include <cudaGL.h>
#include <cuda_gl_interop.h>
#include <nv_node/macros.hpp>
#include <nv_node/utilities/args.hpp>

namespace nv {

// cudaError_t CUDARTAPI cudaGLGetDevices(unsigned int *pCudaDeviceCount, int *pCudaDevices,
// unsigned int cudaDeviceCount, enum cudaGLDeviceList deviceList)
Napi::Value cudaGLGetDevices(CallbackArgs const& info) {
  auto env                   = info.Env();
  uint32_t cu_GL_device_list = info[0];
  uint32_t device_count{};
  std::vector<int> devices{};
  NODE_CUDA_TRY(CUDARTAPI::cudaGLGetDevices(&device_count,
                                            devices.data(),
                                            devices.capacity(),
                                            static_cast<cudaGLDeviceList>(cu_GL_device_list)),
                env);
  return CPPToNapi(info)(devices);
}

// cudaError_t CUDARTAPI cudaGraphicsGLRegisterBuffer(cudaGraphicsResource_t *resource, GLuint
// buffer, unsigned int flags)
Napi::Value cudaGraphicsGLRegisterBuffer(CallbackArgs const& info) {
  auto env       = info.Env();
  GLuint buffer  = info[0];
  uint32_t flags = info[1];
  cudaGraphicsResource_t resource;
  NODE_CUDA_TRY(CUDARTAPI::cudaGraphicsGLRegisterBuffer(&resource, buffer, flags), env);
  return CPPToNapi(info)(resource);
}

// cudaError_t CUDARTAPI cudaGraphicsGLRegisterImage(cudaGraphicsResource_t *resource, GLuint image,
// GLenum target, unsigned int flags)
Napi::Value cudaGraphicsGLRegisterImage(CallbackArgs const& info) {
  auto env       = info.Env();
  GLuint image   = info[0];
  GLenum target  = info[1];
  uint32_t flags = info[2];
  cudaGraphicsResource_t resource;
  NODE_CUDA_TRY(CUDARTAPI::cudaGraphicsGLRegisterImage(&resource, image, target, flags), env);
  return CPPToNapi(info)(resource);
}

// cudaError_t CUDARTAPI cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource)
Napi::Value cudaGraphicsUnregisterResource(CallbackArgs const& info) {
  auto env                        = info.Env();
  cudaGraphicsResource_t resource = info[0];
  NODE_CUDA_TRY(CUDARTAPI::cudaGraphicsUnregisterResource(resource), env);
  return env.Undefined();
}

// cudaError_t CUDARTAPI cudaGraphicsMapResources(int count, cudaGraphicsResource_t *resources,
// cudaStream_t stream = 0)
Napi::Value cudaGraphicsMapResources(CallbackArgs const& info) {
  auto env                                      = info.Env();
  std::vector<cudaGraphicsResource_t> resources = info[0];
  cudaStream_t stream                           = info[1];
  NODE_CUDA_TRY(CUDARTAPI::cudaGraphicsMapResources(resources.size(), resources.data(), stream),
                env);
  return env.Undefined();
}

// cudaError_t CUDARTAPI cudaGraphicsUnmapResources(int count, cudaGraphicsResource_t *resources,
// cudaStream_t stream = 0)
Napi::Value cudaGraphicsUnmapResources(CallbackArgs const& info) {
  auto env                                      = info.Env();
  std::vector<cudaGraphicsResource_t> resources = info[0];
  cudaStream_t stream                           = info[1];
  NODE_CUDA_TRY(CUDARTAPI::cudaGraphicsUnmapResources(resources.size(), resources.data(), stream),
                env);
  return env.Undefined();
}

// cudaError_t CUDARTAPI cudaGraphicsResourceGetMappedPointer(void **devPtr, size_t *size,
// cudaGraphicsResource_t resource)
Napi::Value cudaGraphicsResourceGetMappedPointer(CallbackArgs const& info) {
  return MappedGLMemory::New(info[0]);
}

// cudaError_t CUDARTAPI cudaGraphicsSubResourceGetMappedArray(cudaArray_t *array,
// cudaGraphicsResource_t resource, unsigned int arrayIndex, unsigned int mipLevel)
Napi::Value cudaGraphicsSubResourceGetMappedArray(CallbackArgs const& info) {
  auto env                        = info.Env();
  cudaGraphicsResource_t resource = info[0];
  uint32_t arrayIndex             = info[1];
  uint32_t mipLevel               = info[2];
  cudaArray_t array;
  size_t size{};
  NODE_CUDA_TRY(
    CUDARTAPI::cudaGraphicsSubResourceGetMappedArray(&array, resource, arrayIndex, mipLevel), env);
  uint32_t flags{};
  cudaExtent extent{};
  cudaChannelFormatDesc desc{};
  NODE_CUDA_TRY(CUDARTAPI::cudaArrayGetInfo(&desc, &extent, &flags, array), env);
  return CUDAArray::New(array, extent, desc, flags, array_type::GL);
}

namespace gl {
Napi::Object initModule(Napi::Env env, Napi::Object exports) {
  EXPORT_FUNC(env, exports, "getDevices", nv::cudaGLGetDevices);
  EXPORT_FUNC(env, exports, "registerBuffer", nv::cudaGraphicsGLRegisterBuffer);
  EXPORT_FUNC(env, exports, "registerImage", nv::cudaGraphicsGLRegisterImage);
  EXPORT_FUNC(env, exports, "unregisterResource", nv::cudaGraphicsUnregisterResource);
  EXPORT_FUNC(env, exports, "mapResources", nv::cudaGraphicsMapResources);
  EXPORT_FUNC(env, exports, "unmapResources", nv::cudaGraphicsUnmapResources);
  EXPORT_FUNC(env, exports, "getMappedPointer", nv::cudaGraphicsResourceGetMappedPointer);
  EXPORT_FUNC(env, exports, "getMappedArray", nv::cudaGraphicsSubResourceGetMappedArray);

  auto cudaGraphicsRegisterFlags = Napi::Object::New(env);
  EXPORT_ENUM(env, cudaGraphicsRegisterFlags, "none", CU_GRAPHICS_REGISTER_FLAGS_NONE);
  EXPORT_ENUM(env, cudaGraphicsRegisterFlags, "read_only", CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY);
  EXPORT_ENUM(
    env, cudaGraphicsRegisterFlags, "write_discard", CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);
  EXPORT_PROP(exports, "graphicsRegisterFlags", cudaGraphicsRegisterFlags);

  return exports;
}
}  // namespace gl
}  // namespace nv
