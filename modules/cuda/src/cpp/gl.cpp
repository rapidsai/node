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

#include <GL/gl.h>
#include <cuda.h>
#include <cudaGL.h>
#include <cuda_gl_interop.h>

#include <node_cuda/array.hpp>
#include <node_cuda/buffer.hpp>
#include <node_cuda/casting.hpp>
#include <node_cuda/macros.hpp>

namespace node_cuda {

// cudaError_t CUDARTAPI cudaGLGetDevices(unsigned int *pCudaDeviceCount, int *pCudaDevices,
// unsigned int cudaDeviceCount, enum cudaGLDeviceList deviceList)
Napi::Value cudaGLGetDevices(Napi::CallbackInfo const& info) {
  auto env                   = info.Env();
  uint32_t cu_GL_device_list = FromJS(info[0]);
  uint32_t device_count{};
  std::vector<int> devices{};
  CUDA_TRY(env,
           CUDARTAPI::cudaGLGetDevices(&device_count,
                                       devices.data(),
                                       devices.capacity(),
                                       static_cast<cudaGLDeviceList>(cu_GL_device_list)));
  return ToNapi(env)(devices);
}

// cudaError_t CUDARTAPI cudaGraphicsGLRegisterBuffer(cudaGraphicsResource_t *resource, GLuint
// buffer, unsigned int flags)
Napi::Value cudaGraphicsGLRegisterBuffer(Napi::CallbackInfo const& info) {
  auto env       = info.Env();
  GLuint buffer  = FromJS(info[0]);
  uint32_t flags = FromJS(info[1]);
  cudaGraphicsResource_t resource;
  CUDA_TRY(env, CUDARTAPI::cudaGraphicsGLRegisterBuffer(&resource, buffer, flags));
  return ToNapi(env)(resource);
}

// cudaError_t CUDARTAPI cudaGraphicsGLRegisterImage(cudaGraphicsResource_t *resource, GLuint image,
// GLenum target, unsigned int flags)
Napi::Value cudaGraphicsGLRegisterImage(Napi::CallbackInfo const& info) {
  auto env       = info.Env();
  GLuint image   = FromJS(info[0]);
  GLenum target  = FromJS(info[1]);
  uint32_t flags = FromJS(info[2]);
  cudaGraphicsResource_t resource;
  CUDA_TRY(env, CUDARTAPI::cudaGraphicsGLRegisterImage(&resource, image, target, flags));
  return ToNapi(env)(resource);
}

// cudaError_t CUDARTAPI cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource)
Napi::Value cudaGraphicsUnregisterResource(Napi::CallbackInfo const& info) {
  auto env                        = info.Env();
  cudaGraphicsResource_t resource = FromJS(info[0]);
  CUDA_TRY(env, CUDARTAPI::cudaGraphicsUnregisterResource(resource));
  return env.Undefined();
}

// cudaError_t CUDARTAPI cudaGraphicsMapResources(int count, cudaGraphicsResource_t *resources,
// cudaStream_t stream = 0)
Napi::Value cudaGraphicsMapResources(Napi::CallbackInfo const& info) {
  auto env                                      = info.Env();
  std::vector<cudaGraphicsResource_t> resources = FromJS(info[0]);
  cudaStream_t stream                           = FromJS(info[1]);
  CUDA_TRY(env, CUDARTAPI::cudaGraphicsMapResources(resources.size(), resources.data(), stream));
  return env.Undefined();
}

// cudaError_t CUDARTAPI cudaGraphicsUnmapResources(int count, cudaGraphicsResource_t *resources,
// cudaStream_t stream = 0)
Napi::Value cudaGraphicsUnmapResources(Napi::CallbackInfo const& info) {
  auto env                                      = info.Env();
  std::vector<cudaGraphicsResource_t> resources = FromJS(info[0]);
  cudaStream_t stream                           = FromJS(info[1]);
  CUDA_TRY(env, CUDARTAPI::cudaGraphicsUnmapResources(resources.size(), resources.data(), stream));
  return env.Undefined();
}

// cudaError_t CUDARTAPI cudaGraphicsResourceGetMappedPointer(void **devPtr, size_t *size,
// cudaGraphicsResource_t resource)
Napi::Value cudaGraphicsResourceGetMappedPointer(Napi::CallbackInfo const& info) {
  auto env                        = info.Env();
  cudaGraphicsResource_t resource = FromJS(info[0]);
  void* data;
  size_t size;
  CUDA_TRY(env, CUDARTAPI::cudaGraphicsResourceGetMappedPointer(&data, &size, resource));
  return CUDABuffer::New(data, size, buffer_type::GL);
}

// cudaError_t CUDARTAPI cudaGraphicsSubResourceGetMappedArray(cudaArray_t *array,
// cudaGraphicsResource_t resource, unsigned int arrayIndex, unsigned int mipLevel)
Napi::Value cudaGraphicsSubResourceGetMappedArray(Napi::CallbackInfo const& info) {
  auto env                        = info.Env();
  cudaGraphicsResource_t resource = FromJS(info[0]);
  uint32_t arrayIndex             = FromJS(info[1]);
  uint32_t mipLevel               = FromJS(info[2]);
  cudaArray_t array;
  size_t size{};
  CUDA_TRY(
    env, CUDARTAPI::cudaGraphicsSubResourceGetMappedArray(&array, resource, arrayIndex, mipLevel));
  uint32_t flags{};
  cudaExtent extent{};
  cudaChannelFormatDesc desc{};
  CUDA_TRY(env, CUDARTAPI::cudaArrayGetInfo(&desc, &extent, &flags, array));
  return CUDAArray::New(array, extent, desc, flags, array_type::GL);
}

namespace gl {
Napi::Object initModule(Napi::Env env, Napi::Object exports) {
  EXPORT_FUNC(env, exports, "getDevices", node_cuda::cudaGLGetDevices);
  EXPORT_FUNC(env, exports, "registerBuffer", node_cuda::cudaGraphicsGLRegisterBuffer);
  EXPORT_FUNC(env, exports, "registerImage", node_cuda::cudaGraphicsGLRegisterImage);
  EXPORT_FUNC(env, exports, "unregisterResource", node_cuda::cudaGraphicsUnregisterResource);
  EXPORT_FUNC(env, exports, "mapResources", node_cuda::cudaGraphicsMapResources);
  EXPORT_FUNC(env, exports, "unmapResources", node_cuda::cudaGraphicsUnmapResources);
  EXPORT_FUNC(env, exports, "getMappedPointer", node_cuda::cudaGraphicsResourceGetMappedPointer);
  EXPORT_FUNC(env, exports, "getMappedArray", node_cuda::cudaGraphicsSubResourceGetMappedArray);

  auto cudaGraphicsRegisterFlags = Napi::Object::New(env);
  EXPORT_ENUM(env, cudaGraphicsRegisterFlags, "none", CU_GRAPHICS_REGISTER_FLAGS_NONE);
  EXPORT_ENUM(env, cudaGraphicsRegisterFlags, "read_only", CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY);
  EXPORT_ENUM(
    env, cudaGraphicsRegisterFlags, "write_discard", CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);
  EXPORT_PROP(exports, "graphicsRegisterFlags", cudaGraphicsRegisterFlags);

  return exports;
}
}  // namespace gl
}  // namespace node_cuda
