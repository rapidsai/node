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

#include <node_cuda/casting.hpp>

#include <cuda.h>
#include <cuda_runtime.h>
#include <napi.h>

namespace node_cuda {

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

// CUresult cuInit(unsigned int Flags)
Napi::Value cuInit(Napi::CallbackInfo const& info);

// CUresult cuDriverGetVersion(int* driverVersion);
Napi::Value cuDriverGetVersion(Napi::CallbackInfo const& info);

/**
 *
 *
 * cudaDevice
 *
 *
 */

// cudaError_t CUDARTAPI::cudaChooseDevice(int *device, const struct
// cudaDeviceProp *prop);
Napi::Value cudaChooseDevice(Napi::CallbackInfo const& info);

// cudaError_t CUDARTAPI::cudaGetDeviceCount(int *count);
Napi::Value cudaGetDeviceCount(Napi::CallbackInfo const& info);

// CUresult CUDARTAPI cuDeviceGet(CUdevice *device, int ordinal);
// Napi::Value cuDeviceGet(Napi::CallbackInfo const& info);
Napi::Value cudaChooseDeviceByIndex(Napi::CallbackInfo const& info);

// cudaError_t CUDARTAPI cudaDeviceGetPCIBusId(char *pciBusId, int len, int
// device);
Napi::Value cudaDeviceGetPCIBusId(Napi::CallbackInfo const& info);

// cudaError_t CUDARTAPI cudaDeviceGetByPCIBusId(int *device, const char
// *pciBusId);
Napi::Value cudaDeviceGetByPCIBusId(Napi::CallbackInfo const& info);

// cudaError_t CUDARTAPI cudaGetDevice(CUdevice *device);
Napi::Value cudaGetDevice(Napi::CallbackInfo const& info);

// cudaError_t CUDARTAPI cudaGetDeviceFlags(unsigned int *flags);
Napi::Value cudaGetDeviceFlags(Napi::CallbackInfo const& info);

// cudaError_t CUDARTAPI cudaSetDevice(CUdevice *device);
Napi::Value cudaSetDevice(Napi::CallbackInfo const& info);

// cudaError_t CUDARTAPI cudaSetDeviceFlags(unsigned int flags);
Napi::Value cudaSetDeviceFlags(Napi::CallbackInfo const& info);

// cudaError_t CUDARTAPI cudaGetDeviceProperties(struct cudaDeviceProp *prop,
// int device);
Napi::Value cudaGetDeviceProperties(Napi::CallbackInfo const& info);

// cudaError_t CUDARTAPI cudaDeviceReset();
Napi::Value cudaDeviceReset(Napi::CallbackInfo const& info);

// cudaError_t CUDARTAPI cudaDeviceSynchronize(void);
Napi::Value cudaDeviceSynchronize(Napi::CallbackInfo const& info);

// cudaError_t CUDARTAPI cudaDeviceCanAccessPeer(int *canAccessPeer, int device,
// int peerDevice);
Napi::Value cudaDeviceCanAccessPeer(Napi::CallbackInfo const& info);

// cudaError_t CUDARTAPI cudaDeviceEnablePeerAccess(int peerDevice, unsigned int
// flags);
Napi::Value cudaDeviceEnablePeerAccess(Napi::CallbackInfo const& info);

// cudaError_t CUDARTAPI cudaDeviceDisablePeerAccess(int peerDevice, unsigned
// int flags);
Napi::Value cudaDeviceDisablePeerAccess(Napi::CallbackInfo const& info);

/**
 *
 *
 * cudaGraphics
 *
 *
 */

// CUresult CUDAAPI cuGraphicsGLRegisterBuffer(CUgraphicsResource
// *pCudaResource, GLuint buffer, unsigned int Flags)
Napi::Value cuGraphicsGLRegisterBuffer(Napi::CallbackInfo const& info);

// CUresult cuGraphicsGLRegisterImage(CUgraphicsResource *pCudaResource, GLuint
// image, GLenum target, unsigned int Flags)
Napi::Value cuGraphicsGLRegisterImage(Napi::CallbackInfo const& info);

// CUresult CUDAAPI cuGraphicsUnregisterResource(CUgraphicsResource resource)
Napi::Value cuGraphicsUnregisterResource(Napi::CallbackInfo const& info);

// CUresult CUDAAPI cuGraphicsMapResources(unsigned int count,
// CUgraphicsResource *resources, CUstream hStream)
Napi::Value cuGraphicsMapResources(Napi::CallbackInfo const& info);

// CUresult CUDAAPI cuGraphicsUnmapResources(unsigned int count,
// CUgraphicsResource *resources, CUstream hStream)
Napi::Value cuGraphicsUnapResources(Napi::CallbackInfo const& info);

// CUresult CUDAAPI cuGraphicsResourceGetMappedPointer(CUdeviceptr *pDevPtr,
// size_t *pSize, CUgraphicsResource resource)
Napi::Value cuGraphicsResourceGetMappedPointer(Napi::CallbackInfo const& info);

/**
 *
 *
 * cudaIpc
 *
 *
 */

// cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t *handle, void *devPtr)
Napi::Value cudaIpcGetMemHandle(Napi::CallbackInfo const& info);

// cudaError_t cudaIpcOpenMemHandle(void **devPtr, cudaIpcMemHandle_t handle,
// unsigned int flags)
Napi::Value cudaIpcOpenMemHandle(Napi::CallbackInfo const& info);

// cudaError_t cudaIpcCloseMemHandle(void *devPtr)
Napi::Value cudaIpcCloseMemHandle(Napi::CallbackInfo const& info);

/**
 *
 *
 * cuKernel
 *
 *
 */

// CUresult cuLaunchKernel(CUfunction f,
//                         unsigned int gridDimX, unsigned int gridDimY,
//                         unsigned int gridDimZ, unsigned int blockDimX,
//                         unsigned int blockDimY, unsigned int blockDimZ,
//                         unsigned int sharedMemBytes, CUstream hStream,
//                         void **kernelParams, void ** extra);
Napi::Value cuLaunchKernel(Napi::CallbackInfo const& info);

/**
 *
 *
 * cudaMem
 *
 *
 */

// cudaError_t cudaMalloc(void **devPtr, size_t size);
// Napi::Value cudaMalloc(Napi::CallbackInfo const& info);
Napi::Value cudaMalloc(CallbackArgs const& info);

// cudaError_t cudaFree(void *devPtr);
Napi::Value cudaFree(Napi::CallbackInfo const& info);

// cudaError_t cudaMallocHost(void **ptr, size_t size);
Napi::Value cudaMallocHost(Napi::CallbackInfo const& info);

// cudaError_t cudaFreeHost(void *ptr);
Napi::Value cudaFreeHost(Napi::CallbackInfo const& info);

// cudaError_t cudaHostRegister(void *ptr, size_t size, unsigned int flags);
Napi::Value cudaHostRegister(Napi::CallbackInfo const& info);

// cudaError_t cudaHostUnregister(void *ptr);
Napi::Value cudaHostUnregister(Napi::CallbackInfo const& info);

// cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);
Napi::Value cudaMemcpy(Napi::CallbackInfo const& info);

// cudaError_t cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width,
// size_t height, enum cudaMemcpyKind kind);
Napi::Value cudaMemcpy2D(Napi::CallbackInfo const& info);

// cudaError_t cudaMemcpy2DFromArray(void *dst, size_t dpitch, cudaArray_const_t src, size_t
// wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind);
Napi::Value cudaMemcpy2DFromArray(Napi::CallbackInfo const& info);

// cudaError_t cudaMemset(void *devPtr, int value, size_t count);
Napi::Value cudaMemset(Napi::CallbackInfo const& info);

// cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind,
// cudaStream_t stream);
Napi::Value cudaMemcpyAsync(Napi::CallbackInfo const& info);

// cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream);
Napi::Value cudaMemsetAsync(Napi::CallbackInfo const& info);

// CUresult cudaMemGetInfo(size_t * free, size_t * total);
Napi::Value cudaMemGetInfo(Napi::CallbackInfo const& info);

// CUresult cuPointerGetAttribute(void *data, CUpointer_attribute attribute, CUdeviceptr ptr);
Napi::Value cuPointerGetAttribute(Napi::CallbackInfo const& info);

/**
 *
 *
 * nvrtcProgram
 *
 *
 */
// nvrtcCreateProgram(nvrtcProgram *prog,
//                    const char *src,
//                    const char *name,
//                    int numHeaders,
//                    const char * const *headers,
//                    const char * const *includeNames)
Napi::Value createProgram(Napi::CallbackInfo const& info);

/**
 *
 *
 * cudaStream
 *
 *
 */

// cudaError_t cudaStreamCreate(cudaStream_t *pStream);
Napi::Value cudaStreamCreate(Napi::CallbackInfo const& info);

// cudaError_t cudaStreamDestroy(cudaStream_t stream);
Napi::Value cudaStreamDestroy(Napi::CallbackInfo const& info);

// cudaError_t cudaStreamSynchronize(cudaStream_t stream);
Napi::Value cudaStreamSynchronize(Napi::CallbackInfo const& info);

}  // namespace node_cuda
