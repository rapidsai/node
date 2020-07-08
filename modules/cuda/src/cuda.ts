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

const CUDA = (() => {
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

export { CUDA };

export interface CUDA {
    CUDABuffer: CUDABufferConstructor;
}

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

export type CUdevice = number;
export type CUresult = number;
export type CUstream = number;
export type CUcontext = object;
export type CUfunction = object;
export type CUgraphicsResource = number;
export type CUDAhostptr = ArrayBuffer | ArrayBufferView;
export type CUdeviceptr = ArrayBuffer | ArrayBufferView | CUDABuffer;
export type CUipcMemHandle = Uint8Array;

export type GLImage = number;
export type GLBuffer = number;
export type nvrtcProgram = object;

export const getDriverVersion: () => number = CUDA.getDriverVersion;

export namespace device {
    export const choose: (props?: CUDADeviceProperties) => CUdevice = CUDA.device.choose;
    export const getCount: () => number = CUDA.device.getCount;
    export const getByIndex: (ordinal: number) => CUdevice = CUDA.device.getByIndex;
    export const getPCIBusId: (device: CUdevice) => string = CUDA.device.getPCIBusId;
    export const getByPCIBusId: (pciBusID: string | number) => CUdevice = CUDA.device.getByPCIBusId;
    export const get: () => CUdevice = CUDA.device.get;
    export const getFlags: () => CUDADeviceFlag = CUDA.device.getFlags;
    export const getProperties: (device: CUdevice) => CUDADeviceProperties = CUDA.device.getProperties;
    export const set: (device: CUdevice) => void = CUDA.device.set;
    export const setFlags: (flags: CUDADeviceFlag) => void = CUDA.device.setFlags;
    export const reset: () => void = CUDA.device.reset;
    export const synchronize: () => void = CUDA.device.synchronize;
    export const canAccessPeer: (device: CUdevice, peerDevice: CUdevice) => boolean = CUDA.device.canAccessPeer;
    export const enablePeerAccess: (peerDevice: CUdevice) => boolean = CUDA.device.enablePeerAccess;
    export const disablePeerAccess: (peerDevice: CUdevice) => boolean = CUDA.device.disablePeerAccess;
}

export namespace gl {
    export const getDevices: (list: 0 | 1 | 2) => number[] = CUDA.gl.getDevices;
    export const registerBuffer: (glBuffer: GLBuffer, flags: number) => CUgraphicsResource = CUDA.gl.registerBuffer;
    export const registerImage: (glImage: GLImage, target: number, flags: number) => CUgraphicsResource = CUDA.gl.registerImage;
    export const unregisterResource: (resource: CUgraphicsResource) => void = CUDA.gl.unregisterResource;
    export const mapResources: (resources: CUgraphicsResource[]) => void = CUDA.gl.mapResources;
    export const unmapResources: (resources: CUgraphicsResource[]) => void = CUDA.gl.unmapResources;
    export const getMappedPointer: (resource: CUgraphicsResource) => CUDABuffer = CUDA.gl.getMappedPointer;
}

export namespace ipc {
    export const getMemHandle: (mem: CUdeviceptr) => CUipcMemHandle = CUDA.ipc.getMemHandle;
    export const openMemHandle: (ipcHandle: CUipcMemHandle) => CUDABuffer = CUDA.ipc.openMemHandle;
    export const closeMemHandle: (ipcHandle: CUipcMemHandle) => void = CUDA.ipc.closeMemHandle;
}

export namespace kernel {
    export const launch: (f: CUfunction, gridDims: [number, number, number], blockDims: [number, number, number], sharedMemSize: number, params: any[], extra: any[]) => void = CUDA.kernel.launch;
}

export namespace math {
    export const E = Math.E;
    export const LN10 = Math.LN10;
    export const LN2 = Math.LN2;
    export const LOG2E = Math.LOG2E;
    export const LOG10E = Math.LOG10E;
    export const PI = Math.PI;
    export const SQRT1_2 = Math.SQRT1_2;
    export const SQRT2 = Math.SQRT2;
    export const random = Math.random;
    export const abs   = <T extends number | bigint>(x: T): T => CUDA.math.abs(x);
    export const acos  = <T extends number | bigint>(x: T): T => CUDA.math.acos(x);
    export const asin  = <T extends number | bigint>(x: T): T => CUDA.math.asin(x);
    export const atan  = <T extends number | bigint>(x: T): T => CUDA.math.atan(x);
    export const atan2 = <T extends number | bigint>(y: T, x: T): T => CUDA.math.atan2(y, x);
    export const ceil  = <T extends number | bigint>(x: T): T => CUDA.math.ceil(x);
    export const cos   = <T extends number | bigint>(x: T): T => CUDA.math.cos(x);
    export const exp   = <T extends number | bigint>(x: T): T => CUDA.math.exp(x);
    export const floor = <T extends number | bigint>(x: T): T => CUDA.math.floor(x);
    export const log   = <T extends number | bigint>(x: T): T => CUDA.math.log(x);
    export const max   = <T extends number | bigint>(a: T, b: T): T => CUDA.math.max(a, b);
    export const min   = <T extends number | bigint>(a: T, b: T): T => CUDA.math.min(a, b);
    export const pow   = <T extends number | bigint>(x: T, y: T): T => CUDA.math.pow(x, y);
    export const round = <T extends number | bigint>(x: T): T => CUDA.math.round(x);
    export const sin   = <T extends number | bigint>(x: T): T => CUDA.math.sin(x);
    export const sqrt  = <T extends number | bigint>(x: T): T => CUDA.math.sqrt(x);
    export const tan   = <T extends number | bigint>(x: T): T => CUDA.math.tan(x);
}

export namespace mem {
    export const alloc: (byteLength: number) => CUDABuffer = CUDA.mem.alloc;
    export const free: (mem: CUdeviceptr) => void = CUDA.mem.free;
    
    export const set: (target: CUdeviceptr, offset: number, fillValue: bigint | number, length: number) => void = CUDA.mem.set;
    export const setAsync: (target: CUdeviceptr, offset: number, fillValue: bigint | number, length: number, stream: CUstream) => Promise<void> = CUDA.mem.setAsync;
    
    export const cpy: (target: CUdeviceptr, targetOffset: number, source: CUdeviceptr, sourceOffset: number, length: number) => void = CUDA.mem.cpy;
    export const cpyAsync: (target: CUdeviceptr, targetOffset: number, source: CUdeviceptr, sourceOffset: number, length: number, stream?: CUstream) => Promise<void> = CUDA.mem.cpyAsync;
    
    export const allocHost: (byteLength: number, flags?: CUDAMemHostAllocFlag) => CUDABuffer = CUDA.mem.allocHost;
    export const freeHost: (mem: CUDAhostptr) => void = CUDA.mem.freeHost;
    export const hostRegister: (target: CUDAhostptr, flags?: CUDAMemHostRegisterFlag) => void = CUDA.mem.hostRegister;
    export const hostUnregister: (target: CUDAhostptr) => void = CUDA.mem.hostUnregister;
    
    export const getInfo: () => { free: number, total: number } = CUDA.mem.getInfo;
    export const getPointerAttribute: (mem: CUdeviceptr | CUDAhostptr, attr: CUDAPointerAttribute) => any = CUDA.mem.getPointerAttribute;
}

export namespace program {
    export const create: (source: string, name: string, headers?: string[], includes?: string[]) => nvrtcProgram = CUDA.program.create;
}

export namespace stream {
    export const create: () => CUstream = CUDA.stream.create;
    export const destroy: (stream: CUstream) => void = CUDA.stream.destroy;
    export const synchronize: (stream: CUstream) => void = CUDA.stream.synchronize;
}

export const VERSION: number = CUDA.VERSION;
export const IPC_HANDLE_SIZE: number = CUDA.IPC_HANDLE_SIZE;

export enum CUDAContextFlag {
    MAP_HOST = CUDA.CTX_MAP_HOST,
    SCHED_AUTO = CUDA.CTX_SCHED_AUTO,
    SCHED_SPIN = CUDA.CTX_SCHED_SPIN,
    SCHED_YIELD = CUDA.CTX_SCHED_YIELD,
    SCHED_BLOCKING_SYNC = CUDA.CTX_SCHED_BLOCKING_SYNC,
    LMEM_RESIZE_TO_MAX = CUDA.CTX_LMEM_RESIZE_TO_MAX
}

export enum CUDADeviceFlag {
    scheduleAuto = CUDA.device.flags.scheduleAuto,
    scheduleSpin = CUDA.device.flags.scheduleSpin,
    scheduleYield = CUDA.device.flags.scheduleYield,
    scheduleBlockingSync = CUDA.device.flags.scheduleBlockingSync,
    mapHost = CUDA.device.flags.mapHost,
    lmemResizeToMax = CUDA.device.flags.lmemResizeToMax,
};

export interface CUDADeviceProperties {
    /** ASCII string identifying device */
    name: string;
    /** 16-byte unique identifier */
    uuid: ArrayBuffer;
    /** Global memory available on device in bytes */
    totalGlobalMem: number;
    /** Shared memory available per block in bytes */
    sharedMemPerBlock: number;
    /** 32-bit registers available per block */
    regsPerBlock: number;
    /** Warp size in threads */
    warpSize: number;
    /** Maximum pitch in bytes allowed by memory copies */
    memPitch: number;
    /** Maximum number of threads per block */
    maxThreadsPerBlock: number;
    /** Maximum size of each dimension of a block */
    maxThreadsDim: ReadonlyArray<number>;
    /** Maximum size of each dimension of a grid */
    maxGridSize: ReadonlyArray<number>;
    /** Clock frequency in kilohertz */
    clockRate: number;
    /** Constant memory available on device in bytes */
    totalConstMem: number;
    /** Major compute capability */
    major: number;
    /** Minor compute capability */
    minor: number;
    /** Alignment requirement for textures */
    textureAlignment: number;
    /** Pitch alignment requirement for texture references bound to pitched memory */
    texturePitchAlignment: number;
    /** Device can concurrently copy memory and execute a kernel. Deprecated. Use instead asyncEngineCount. */
    deviceOverlap: number;
    /** Number of multiprocessors on device */
    multiProcessorCount: number;
    /** Specified whether there is a run time limit on kernels */
    kernelExecTimeoutEnabled: number;
    /** Device is integrated as opposed to discrete */
    integrated: number;
    /** Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer */
    canMapHostMemory: number;
    /** Compute mode (See ::cudaComputeMode) */
    computeMode: number;
    /** Maximum 1D texture size */
    maxTexture1D: number;
    /** Maximum 1D mipmapped texture size */
    maxTexture1DMipmap: number;
    /** Maximum size for 1D textures bound to linear memory */
    maxTexture1DLinear: number;
    /** Maximum 2D texture dimensions */
    maxTexture2D: ReadonlyArray<number>;
    /** Maximum 2D mipmapped texture dimensions */
    maxTexture2DMipmap: ReadonlyArray<number>;
    /** Maximum dimensions (width, height, pitch) for 2D textures bound to pitched memory */
    maxTexture2DLinear: ReadonlyArray<number>;
    /** Maximum 2D texture dimensions if texture gather operations have to be performed */
    maxTexture2DGather: ReadonlyArray<number>;
    /** Maximum 3D texture dimensions */
    maxTexture3D: ReadonlyArray<number>;
    /** Maximum alternate 3D texture dimensions */
    maxTexture3DAlt: ReadonlyArray<number>;
    /** Maximum Cubemap texture dimensions */
    maxTextureCubemap: number;
    /** Maximum 1D layered texture dimensions */
    maxTexture1DLayered: ReadonlyArray<number>;
    /** Maximum 2D layered texture dimensions */
    maxTexture2DLayered: ReadonlyArray<number>;
    /** Maximum Cubemap layered texture dimensions */
    maxTextureCubemapLayered: ReadonlyArray<number>;
    /** Maximum 1D surface size */
    maxSurface1D: number;
    /** Maximum 2D surface dimensions */
    maxSurface2D: ReadonlyArray<number>;
    /** Maximum 3D surface dimensions */
    maxSurface3D: ReadonlyArray<number>;
    /** Maximum 1D layered surface dimensions */
    maxSurface1DLayered: ReadonlyArray<number>;
    /** Maximum 2D layered surface dimensions */
    maxSurface2DLayered: ReadonlyArray<number>;
    /** Maximum Cubemap surface dimensions */
    maxSurfaceCubemap: number;
    /** Maximum Cubemap layered surface dimensions */
    maxSurfaceCubemapLayered: ReadonlyArray<number>;
    /** Alignment requirements for surfaces */
    surfaceAlignment: number;
    /** Device can possibly execute multiple kernels concurrently */
    concurrentKernels: number;
    /** Device has ECC support enabled */
    ECCEnabled: number;
    /** PCI bus ID of the device */
    pciBusID: number;
    /** PCI device ID of the device */
    pciDeviceID: number;
    /** PCI domain ID of the device */
    pciDomainID: number;
    /** 1 if device is a Tesla device using TCC driver, 0 otherwise */
    tccDriver: number;
    /** Number of asynchronous engines */
    asyncEngineCount: number;
    /** Device shares a unified address space with the host */
    unifiedAddressing: number;
    /** Peak memory clock frequency in kilohertz */
    memoryClockRate: number;
    /** Global memory bus width in bits */
    memoryBusWidth: number;
    /** Size of L2 cache in bytes */
    l2CacheSize: number;
    /** Maximum resident threads per multiprocessor */
    maxThreadsPerMultiProcessor: number;
    /** Device supports stream priorities */
    streamPrioritiesSupported: number;
    /** Device supports caching globals in L1 */
    globalL1CacheSupported: number;
    /** Device supports caching locals in L1 */
    localL1CacheSupported: number;
    /** Shared memory available per multiprocessor in bytes */
    sharedMemPerMultiprocessor: number;
    /** 32-bit registers available per multiprocessor */
    regsPerMultiprocessor: number;
    /** Device supports allocating managed memory on this system */
    managedMemory: number;
    /** Device is on a multi-GPU board */
    isMultiGpuBoard: number;
    /** Unique identifier for a group of devices on the same multi-GPU board */
    multiGpuBoardGroupID: number;
    /** Link between the device and the host supports native atomic operations */
    hostNativeAtomicSupported: number;
    /** Ratio of single precision performance (in floating-point operations per second) to double precision performance */
    singleToDoublePrecisionPerfRatio: number;
    /** Device supports coherently accessing pageable memory without calling cudaHostRegister on it */
    pageableMemoryAccess: number;
    /** Device can coherently access managed memory concurrently with the CPU */
    concurrentManagedAccess: number;
    /** Device supports Compute Preemption */
    computePreemptionSupported: number;
    /** Device can access host registered memory at the same virtual address as the CPU */
    canUseHostPointerForRegisteredMem: number;
    /** Device supports launching cooperative kernels via ::cudaLaunchCooperativeKernel */
    cooperativeLaunch: number;
    /** Device can participate in cooperative kernels launched via ::cudaLaunchCooperativeKernelMultiDevice */
    cooperativeMultiDeviceLaunch: number;
    /** Per device maximum shared memory per block usable by special opt in */
    sharedMemPerBlockOptin: number;
    /** Device accesses pageable memory via the host's page tables */
    pageableMemoryAccessUsesHostPageTables: number;
    /** Host can directly access managed memory on the device without migration. */
    directManagedMemAccessFromHost: number;
}

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
