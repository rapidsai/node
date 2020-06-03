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

#include <cuda_runtime.h>
#include <cstdlib>
#include <visit_struct/visit_struct.hpp>

VISITABLE_STRUCT(CUDARTAPI::cudaPointerAttributes, type, device, devicePointer, hostPointer);

static_assert(visit_struct::traits::is_visitable<CUDARTAPI::cudaPointerAttributes>::value, "");

VISITABLE_STRUCT(CUDARTAPI::cudaDeviceProp,
                 name,
                 //  uuid,
                 totalGlobalMem,
                 sharedMemPerBlock,
                 regsPerBlock,
                 warpSize,
                 memPitch,
                 maxThreadsPerBlock,
                 maxThreadsDim,
                 maxGridSize,
                 clockRate,
                 totalConstMem,
                 major,
                 minor,
                 textureAlignment,
                 texturePitchAlignment,
                 deviceOverlap,
                 multiProcessorCount,
                 kernelExecTimeoutEnabled,
                 integrated,
                 canMapHostMemory,
                 computeMode,
                 maxTexture1D,
                 maxTexture1DMipmap,
                 maxTexture1DLinear,
                 maxTexture2D,
                 maxTexture2DMipmap,
                 maxTexture2DLinear,
                 maxTexture2DGather,
                 maxTexture3D,
                 maxTexture3DAlt,
                 maxTextureCubemap,
                 maxTexture1DLayered,
                 maxTexture2DLayered,
                 maxTextureCubemapLayered,
                 maxSurface1D,
                 maxSurface2D,
                 maxSurface3D,
                 maxSurface1DLayered,
                 maxSurface2DLayered,
                 maxSurfaceCubemap,
                 maxSurfaceCubemapLayered,
                 surfaceAlignment,
                 concurrentKernels,
                 ECCEnabled,
                 pciBusID,
                 pciDeviceID,
                 pciDomainID,
                 tccDriver,
                 asyncEngineCount,
                 unifiedAddressing,
                 memoryClockRate,
                 memoryBusWidth,
                 l2CacheSize,
                 maxThreadsPerMultiProcessor,
                 streamPrioritiesSupported,
                 globalL1CacheSupported,
                 localL1CacheSupported,
                 sharedMemPerMultiprocessor,
                 regsPerMultiprocessor,
                 managedMemory,
                 isMultiGpuBoard,
                 multiGpuBoardGroupID,
                 hostNativeAtomicSupported,
                 singleToDoublePrecisionPerfRatio,
                 pageableMemoryAccess,
                 concurrentManagedAccess,
                 computePreemptionSupported,
                 canUseHostPointerForRegisteredMem,
                 cooperativeLaunch,
                 cooperativeMultiDeviceLaunch,
                 sharedMemPerBlockOptin,
                 pageableMemoryAccessUsesHostPageTables,
                 directManagedMemAccessFromHost);

static_assert(visit_struct::traits::is_visitable<CUDARTAPI::cudaDeviceProp>::value, "");
