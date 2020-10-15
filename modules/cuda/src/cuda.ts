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

export * from './addon';

// export { CUDA };

// export type CUdevice = number;
// export type CUresult = number;
// export type CUstream = number;
// export type CUcontext = object;
// export type CUfunction = object;
// export type CUgraphicsResource = number;
// export type CUDAhostptr = ArrayBuffer | ArrayBufferView;
// export type CUdeviceptr = ArrayBuffer | ArrayBufferView | CUDABuffer;
// export type CUarrayptr = number;
// export type CUipcMemHandle = Uint8Array;

// export type GLImage = number;
// export type GLBuffer = number;
// export type nvrtcProgram = object;

// export const getDriverVersion: () => number = CUDA.getDriverVersion;

// export namespace gl {
//     export const getDevices: (list: 0 | 1 | 2) => number[] = CUDA.gl.getDevices;
//     export const registerBuffer: (glBuffer: GLBuffer, flags: number) => CUgraphicsResource = CUDA.gl.registerBuffer;
//     export const registerImage: (glImage: GLImage, target: number, flags: number) => CUgraphicsResource = CUDA.gl.registerImage;
//     export const unregisterResource: (resource: CUgraphicsResource) => void = CUDA.gl.unregisterResource;
//     export const mapResources: (resources: CUgraphicsResource[]) => void = CUDA.gl.mapResources;
//     export const unmapResources: (resources: CUgraphicsResource[]) => void = CUDA.gl.unmapResources;
//     export const getMappedArray: (resource: CUgraphicsResource) => CUDAArray = CUDA.gl.getMappedArray;
//     export const getMappedPointer: (resource: CUgraphicsResource) => CUDABuffer = CUDA.gl.getMappedPointer;
// }

// export namespace ipc {
//     export const getMemHandle: (mem: CUdeviceptr) => CUipcMemHandle = CUDA.ipc.getMemHandle;
//     export const openMemHandle: (ipcHandle: CUipcMemHandle) => CUDABuffer = CUDA.ipc.openMemHandle;
//     export const closeMemHandle: (ipcHandle: CUipcMemHandle) => void = CUDA.ipc.closeMemHandle;
// }

// export namespace kernel {
//     export const launch: (f: CUfunction, gridDims: [number, number, number], blockDims: [number, number, number], sharedMemSize: number, params: any[], extra: any[]) => void = CUDA.kernel.launch;
// }

// export namespace math {
//     export const E = Math.E;
//     export const LN10 = Math.LN10;
//     export const LN2 = Math.LN2;
//     export const LOG2E = Math.LOG2E;
//     export const LOG10E = Math.LOG10E;
//     export const PI = Math.PI;
//     export const SQRT1_2 = Math.SQRT1_2;
//     export const SQRT2 = Math.SQRT2;
//     export const random = Math.random;
//     export const abs   = <T extends number | bigint>(x: T): T => CUDA.math.abs(x);
//     export const acos  = <T extends number | bigint>(x: T): T => CUDA.math.acos(x);
//     export const asin  = <T extends number | bigint>(x: T): T => CUDA.math.asin(x);
//     export const atan  = <T extends number | bigint>(x: T): T => CUDA.math.atan(x);
//     export const atan2 = <T extends number | bigint>(y: T, x: T): T => CUDA.math.atan2(y, x);
//     export const ceil  = <T extends number | bigint>(x: T): T => CUDA.math.ceil(x);
//     export const cos   = <T extends number | bigint>(x: T): T => CUDA.math.cos(x);
//     export const exp   = <T extends number | bigint>(x: T): T => CUDA.math.exp(x);
//     export const floor = <T extends number | bigint>(x: T): T => CUDA.math.floor(x);
//     export const log   = <T extends number | bigint>(x: T): T => CUDA.math.log(x);
//     export const max   = <T extends number | bigint>(a: T, b: T): T => CUDA.math.max(a, b);
//     export const min   = <T extends number | bigint>(a: T, b: T): T => CUDA.math.min(a, b);
//     export const pow   = <T extends number | bigint>(x: T, y: T): T => CUDA.math.pow(x, y);
//     export const round = <T extends number | bigint>(x: T): T => CUDA.math.round(x);
//     export const sin   = <T extends number | bigint>(x: T): T => CUDA.math.sin(x);
//     export const sqrt  = <T extends number | bigint>(x: T): T => CUDA.math.sqrt(x);
//     export const tan   = <T extends number | bigint>(x: T): T => CUDA.math.tan(x);
// }

// export namespace mem {
//     export const alloc: (byteLength: number) => CUDABuffer = CUDA.mem.alloc;
//     export const free: (mem: CUdeviceptr) => void = CUDA.mem.free;
    
//     export const set: (target: CUdeviceptr, offset: number, fillValue: bigint | number, length: number) => void = CUDA.mem.set;
//     export const setAsync: (target: CUdeviceptr, offset: number, fillValue: bigint | number, length: number, stream: CUstream) => Promise<void> = CUDA.mem.setAsync;
    
//     export const cpy: (target: CUdeviceptr, targetOffset: number, source: CUdeviceptr, sourceOffset: number, length: number) => void = CUDA.mem.cpy;
//     export const cpy2D: (target: CUarrayptr, targetPitch: number, source: CUarrayptr, sourcePitch: number, width: number, height: number) => void = CUDA.mem.cpy2D;
//     export const cpyAsync: (target: CUdeviceptr, targetOffset: number, source: CUdeviceptr, sourceOffset: number, length: number, stream?: CUstream) => Promise<void> = CUDA.mem.cpyAsync;
//     export const cpy2DFromArray: (target: CUdeviceptr, targetPitch: number, source: CUarrayptr, x: number, y: number, width: number, height: number) => void = CUDA.mem.cpy2DFromArray;
    
//     export const allocHost: (byteLength: number, flags?: CUDAMemHostAllocFlag) => CUDABuffer = CUDA.mem.allocHost;
//     export const freeHost: (mem: CUDAhostptr) => void = CUDA.mem.freeHost;
//     export const hostRegister: (target: CUDAhostptr, flags?: CUDAMemHostRegisterFlag) => void = CUDA.mem.hostRegister;
//     export const hostUnregister: (target: CUDAhostptr) => void = CUDA.mem.hostUnregister;
    
//     export const getInfo: () => { free: number, total: number } = CUDA.mem.getInfo;
//     export const getPointerAttribute: (mem: CUdeviceptr | CUDAhostptr, attr: CUDAPointerAttribute) => any = CUDA.mem.getPointerAttribute;
// }

// export namespace program {
//     export const create: (source: string, name: string, headers?: string[], includes?: string[]) => nvrtcProgram = CUDA.program.create;
// }

// export namespace stream {
//     export const create: () => CUstream = CUDA.stream.create;
//     export const destroy: (stream: CUstream) => void = CUDA.stream.destroy;
//     export const synchronize: (stream: CUstream) => void = CUDA.stream.synchronize;
// }

// export const VERSION: number = CUDA.VERSION;
// export const IPC_HANDLE_SIZE: number = CUDA.IPC_HANDLE_SIZE;

// export enum CUDAContextFlag {
//     MAP_HOST = CUDA.CTX_MAP_HOST,
//     SCHED_AUTO = CUDA.CTX_SCHED_AUTO,
//     SCHED_SPIN = CUDA.CTX_SCHED_SPIN,
//     SCHED_YIELD = CUDA.CTX_SCHED_YIELD,
//     SCHED_BLOCKING_SYNC = CUDA.CTX_SCHED_BLOCKING_SYNC,
//     LMEM_RESIZE_TO_MAX = CUDA.CTX_LMEM_RESIZE_TO_MAX
// }

// export enum CUDAGraphicsRegisterFlag {
//     NONE = CUDA.gl.graphicsRegisterFlags.none,
//     READ_ONLY = CUDA.gl.graphicsRegisterFlags.read_only,
//     WRITE_DISCARD = CUDA.gl.graphicsRegisterFlags.write_discard,
// }

// export enum CUDAMemHostAllocFlag {
//     DEFAULT = CUDA.mem.hostAllocFlags.default,
//     PORTABLE = CUDA.mem.hostAllocFlags.portable,
//     DEVICEMAP = CUDA.mem.hostAllocFlags.devicemap,
//     WRITECOMBINED = CUDA.mem.hostAllocFlags.writeCombined,
// }

// export enum CUDAMemHostRegisterFlag {
//     DEFAULT = CUDA.mem.hostRegisterFlags.default,
//     PORTABLE = CUDA.mem.hostRegisterFlags.portable,
//     DEVICEMAP = CUDA.mem.hostRegisterFlags.devicemap,
//     IOMEMORY = CUDA.mem.hostRegisterFlags.ioMemory,
// }

// export enum CUDAMemoryType {
//     UNREGISTERED = CUDA.mem.memoryTypes.unregistered,
//     HOST = CUDA.mem.memoryTypes.host,
//     DEVICE = CUDA.mem.memoryTypes.device,
//     MANAGED = CUDA.mem.memoryTypes.managed,
// }

// export enum CUDAPointerAttribute {
//     CONTEXT = CUDA.mem.pointerAttributes.context,
//     MEMORY_TYPE = CUDA.mem.pointerAttributes.memory_type,
//     DEVICE_POINTER = CUDA.mem.pointerAttributes.device_pointer,
//     HOST_POINTER = CUDA.mem.pointerAttributes.host_pointer,
//     // P2P_TOKENS = CUDA.mem.pointerAttributes.p2p_tokens,
//     SYNC_MEMOPS = CUDA.mem.pointerAttributes.sync_memops,
//     BUFFER_ID = CUDA.mem.pointerAttributes.buffer_id,
//     IS_MANAGED = CUDA.mem.pointerAttributes.is_managed,
//     DEVICE_ORDINAL = CUDA.mem.pointerAttributes.device_ordinal,
// }
