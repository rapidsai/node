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

import {loadNativeModule} from '@rapidsai/core';

import {MemoryData} from './interfaces';
import {MappedGLMemory} from './memory';

/** @ignore */
export const CUDA = loadNativeModule<any>(module, 'node_cuda');
/** @ignore */
export default CUDA;

/** @ignore */
export type CUdevice = number;
/** @ignore */
export type CUresult = number;
/** @ignore */
export type CUstream = number;
/** @ignore */
export type CUcontext = Record<string, unknown>;
/** @ignore */
export type CUfunction = Record<string, unknown>;
/** @ignore */
export type CUgraphicsResource = number;
/** @ignore */
export type CUDAhostptr = ArrayBuffer|ArrayBufferView;
/** @ignore */
export type CUdeviceptr = ArrayBuffer|ArrayBufferView|MemoryData;
/** @ignore */
export type CUarrayptr = number;
/** @ignore */
export type CUipcMemHandle = Uint8Array;
/** @ignore */
export type GLImage = number;
/** @ignore */
export type GLBuffer = number;
/** @ignore */
export type nvrtcProgram = Record<string, unknown>;

/** @ignore */
// eslint-disable-next-line @typescript-eslint/no-redeclare
export interface CUDA {
  readonly VERSION: number;
  readonly IPC_HANDLE_SIZE: number;

  readonly CUDAArray: CUDAArrayConstructor;

  getDriverVersion(): number;

  readonly gl: {
    getDevices(list: 0|1|2): number[];
    registerBuffer(glBuffer: GLBuffer, flags: number): CUgraphicsResource;
    registerImage(glImage: GLImage, target: number, flags: number): CUgraphicsResource;
    unregisterResource(resource: CUgraphicsResource): void;
    mapResources(resources: CUgraphicsResource[]): void;
    unmapResources(resources: CUgraphicsResource[]): void;
    getMappedArray(resource: CUgraphicsResource): CUDAArray;
    getMappedPointer(resource: CUgraphicsResource): MappedGLMemory;

    readonly graphicsRegisterFlags:
               {readonly none: number; readonly read_only: number; readonly write_discard: number;}
  };

  readonly driver: {
        cuPointerGetAttribute(mem: CUdeviceptr | CUDAhostptr, attr: CUDAPointerAttribute): any;
        readonly PointerAttributes: {
            readonly context: number;
            readonly memory_type: number;
            readonly device_pointer: number;
            readonly host_pointer: number;
            // readonly p2p_tokens: number;
            readonly sync_memops: number;
            readonly buffer_id: number;
            readonly is_managed: number;
            readonly device_ordinal: number;
        };
    };

  readonly runtime: {
    cudaMemGetInfo(): {free: number, total: number};
    cudaMemset(target: MemoryData, value: number, count: number, stream?: number) : void;
    cudaMemcpy(target: MemoryData, source: MemoryData, count: number, stream?: number) : void;
  }

  readonly DeviceFlags: {
    readonly scheduleAuto: number; //
    readonly scheduleSpin : number;
    readonly scheduleYield : number;
    readonly scheduleBlockingSync : number;
    readonly mapHost : number;
    readonly lmemResizeToMax : number;
  }
}

/** @ignore */
export interface CUDAArrayConstructor {
  readonly prototype: CUDAArray;
  new(): CUDAArray;
}

/** @ignore */
export interface CUDAArray {
  readonly ary: number;
  readonly byteLength: number;
  readonly bytesPerElement: number;
  readonly width: number;
  readonly height: number;
  readonly depth: number;
  readonly channelFormatX: number;
  readonly channelFormatY: number;
  readonly channelFormatZ: number;
  readonly channelFormatW: number;
  readonly channelFormatKind: number;
}

/** @ignore */
// eslint-disable-next-line @typescript-eslint/no-redeclare
export declare const CUDAArray: CUDAArrayConstructor;

/** @ignore */
export enum CUDAGraphicsRegisterFlag
{
  NONE          = CUDA.gl.graphicsRegisterFlags.none,
  READ_ONLY     = CUDA.gl.graphicsRegisterFlags.read_only,
  WRITE_DISCARD = CUDA.gl.graphicsRegisterFlags.write_discard,
}

// /** @ignore */
// export enum CUDAMemHostAllocFlag {
//     DEFAULT = CUDA.mem.hostAllocFlags.default,
//     PORTABLE = CUDA.mem.hostAllocFlags.portable,
//     DEVICEMAP = CUDA.mem.hostAllocFlags.devicemap,
//     WRITECOMBINED = CUDA.mem.hostAllocFlags.writeCombined,
// }

// /** @ignore */
// export enum CUDAMemHostRegisterFlag {
//     DEFAULT = CUDA.mem.hostRegisterFlags.default,
//     PORTABLE = CUDA.mem.hostRegisterFlags.portable,
//     DEVICEMAP = CUDA.mem.hostRegisterFlags.devicemap,
//     IOMEMORY = CUDA.mem.hostRegisterFlags.ioMemory,
// }

// /** @ignore */
// export enum CUDAMemoryType {
//     UNREGISTERED = CUDA.mem.memoryTypes.unregistered,
//     HOST = CUDA.mem.memoryTypes.host,
//     DEVICE = CUDA.mem.memoryTypes.device,
//     MANAGED = CUDA.mem.memoryTypes.managed,
// }

/** @ignore */
export enum CUDAPointerAttribute
{
  CONTEXT        = CUDA.driver.PointerAttributes.context,
  MEMORY_TYPE    = CUDA.driver.PointerAttributes.memory_type,
  DEVICE_POINTER = CUDA.driver.PointerAttributes.device_pointer,
  HOST_POINTER   = CUDA.driver.PointerAttributes.host_pointer,
  // P2P_TOKENS = CUDA.driver.PointerAttributes.p2p_tokens,
  SYNC_MEMOPS    = CUDA.driver.PointerAttributes.sync_memops,
  BUFFER_ID      = CUDA.driver.PointerAttributes.buffer_id,
  IS_MANAGED     = CUDA.driver.PointerAttributes.is_managed,
  DEVICE_ORDINAL = CUDA.driver.PointerAttributes.device_ordinal,
}
