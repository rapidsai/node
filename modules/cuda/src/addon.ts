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

import { MemoryData } from './interfaces';

export const CUDA = (() => {
    let CUDA: any, types = ['Release'];
    if (process.env.NODE_DEBUG !== undefined || process.env.NODE_ENV === 'debug') {
        types.push('Debug');
    }
    for (let type; type = types.pop();) {
        try {
            if (CUDA = require(`../${type}/node_cuda.node`)) {
                break;
            }
        } catch (e) { console.error(e); continue; }
    }
    if (CUDA) return CUDA.init();
    throw new Error('node_cuda not found');
})();

export default CUDA;

export type CUdevice = number;
export type CUresult = number;
export type CUstream = number;
export type CUcontext = object;
export type CUfunction = object;
export type CUgraphicsResource = number;
export type CUDAhostptr = ArrayBuffer | ArrayBufferView;
export type CUdeviceptr = ArrayBuffer | ArrayBufferView | CUDABuffer;
export type CUarrayptr = number;
export type CUipcMemHandle = Uint8Array;

export type GLImage = number;
export type GLBuffer = number;
export type nvrtcProgram = object;

export interface CUDA {
    readonly VERSION: number;
    readonly IPC_HANDLE_SIZE: number;

    readonly CUDAArray: CUDAArrayConstructor;
    readonly CUDABuffer: CUDABufferConstructor;

    getDriverVersion(): number;

    readonly gl: {
        getDevices(list: 0 | 1 | 2): number[];
        registerBuffer(glBuffer: GLBuffer, flags: number): CUgraphicsResource;
        registerImage(glImage: GLImage, target: number, flags: number): CUgraphicsResource;
        unregisterResource(resource: CUgraphicsResource): void;
        mapResources(resources: CUgraphicsResource[]): void;
        unmapResources(resources: CUgraphicsResource[]): void;
        getMappedArray(resource: CUgraphicsResource): CUDAArray;
        getMappedPointer(resource: CUgraphicsResource): CUDABuffer;

        readonly graphicsRegisterFlags: {
            readonly none: number;
            readonly read_only: number;
            readonly write_discard: number;
        }
    };

    readonly ipc: {
        getMemHandle(mem: CUdeviceptr): CUipcMemHandle;
        openMemHandle(ipcHandle: CUipcMemHandle): CUDABuffer;
        closeMemHandle(ipcHandle: CUipcMemHandle): void;
    };

    readonly mem: {
        alloc(byteLength: number): CUDABuffer;
        free(mem: CUdeviceptr): void;
        set(target: CUdeviceptr, offset: number, fillValue: bigint | number, length: number): void;
        setAsync(target: CUdeviceptr, offset: number, fillValue: bigint | number, length: number, stream: CUstream): Promise<void>;
        cpy(target: CUdeviceptr, targetOffset: number, source: CUdeviceptr, sourceOffset: number, length: number): void;
        cpy2D(target: CUarrayptr, targetPitch: number, source: CUarrayptr, sourcePitch: number, width: number, height: number): void;
        cpyAsync(target: CUdeviceptr, targetOffset: number, source: CUdeviceptr, sourceOffset: number, length: number, stream?: CUstream): Promise<void>;
        cpy2DFromArray(target: CUdeviceptr, targetPitch: number, source: CUarrayptr, x: number, y: number, width: number, height: number): void;
        allocHost(byteLength: number, flags?: CUDAMemHostAllocFlag): CUDABuffer;
        freeHost(mem: CUDAhostptr): void;
        hostRegister(target: CUDAhostptr, flags?: CUDAMemHostRegisterFlag): void;
        hostUnregister(target: CUDAhostptr): void;
        getInfo(): { free: number, total: number };
        getPointerAttribute(mem: CUdeviceptr | CUDAhostptr, attr: CUDAPointerAttribute): any;

        readonly hostAllocFlags: {
            readonly default: number;
            readonly portable: number;
            readonly devicemap: number;
            readonly writeCombined: number;
        };

        readonly hostRegisterFlags: {
            readonly default: number;
            readonly portable: number;
            readonly devicemap: number;
            readonly ioMemory: number;
        };

        readonly memoryTypes: {
            readonly unregistered: number;
            readonly host: number;
            readonly device: number;
            readonly managed: number;
        };

        readonly pointerAttributes: {
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
        cudaMemset(target: MemoryData, value: number, count: number, stream?: number): void;
        cudaMemcpy(target: MemoryData, source: MemoryData, count: number, stream?: number): void;
    };

    readonly DeviceFlag: {
        readonly scheduleAuto: number;
        readonly scheduleSpin: number;
        readonly scheduleYield: number;
        readonly scheduleBlockingSync: number;
        readonly mapHost: number;
        readonly lmemResizeToMax: number;
    }
}

export interface CUDAArrayConstructor {
    readonly prototype: CUDAArray;
    new(): CUDAArray;
}

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

export declare var CUDAArray: CUDAArrayConstructor;

export interface CUDABufferConstructor {
    readonly prototype: CUDABuffer;
    new(): CUDABuffer;
}

export interface CUDABuffer {
    readonly ptr: number;
    readonly byteLength: number;
    slice(begin: number, end?: number): CUDABuffer;
}

export declare var CUDABuffer: CUDABufferConstructor;

export enum CUDAGraphicsRegisterFlag {
    NONE = CUDA.gl.graphicsRegisterFlags.none,
    READ_ONLY = CUDA.gl.graphicsRegisterFlags.read_only,
    WRITE_DISCARD = CUDA.gl.graphicsRegisterFlags.write_discard,
}

export enum CUDAMemHostAllocFlag {
    DEFAULT = CUDA.mem.hostAllocFlags.default,
    PORTABLE = CUDA.mem.hostAllocFlags.portable,
    DEVICEMAP = CUDA.mem.hostAllocFlags.devicemap,
    WRITECOMBINED = CUDA.mem.hostAllocFlags.writeCombined,
}

export enum CUDAMemHostRegisterFlag {
    DEFAULT = CUDA.mem.hostRegisterFlags.default,
    PORTABLE = CUDA.mem.hostRegisterFlags.portable,
    DEVICEMAP = CUDA.mem.hostRegisterFlags.devicemap,
    IOMEMORY = CUDA.mem.hostRegisterFlags.ioMemory,
}

export enum CUDAMemoryType {
    UNREGISTERED = CUDA.mem.memoryTypes.unregistered,
    HOST = CUDA.mem.memoryTypes.host,
    DEVICE = CUDA.mem.memoryTypes.device,
    MANAGED = CUDA.mem.memoryTypes.managed,
}

export enum CUDAPointerAttribute {
    CONTEXT = CUDA.mem.pointerAttributes.context,
    MEMORY_TYPE = CUDA.mem.pointerAttributes.memory_type,
    DEVICE_POINTER = CUDA.mem.pointerAttributes.device_pointer,
    HOST_POINTER = CUDA.mem.pointerAttributes.host_pointer,
    // P2P_TOKENS = CUDA.mem.pointerAttributes.p2p_tokens,
    SYNC_MEMOPS = CUDA.mem.pointerAttributes.sync_memops,
    BUFFER_ID = CUDA.mem.pointerAttributes.buffer_id,
    IS_MANAGED = CUDA.mem.pointerAttributes.is_managed,
    DEVICE_ORDINAL = CUDA.mem.pointerAttributes.device_ordinal,
}
