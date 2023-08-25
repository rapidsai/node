// Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

/* eslint-disable @typescript-eslint/no-namespace */

import {MemoryData} from './interfaces';

/** @ignore */
export declare const _cpp_exports: any;

export declare const VERSION: number;
export declare const IPC_HANDLE_SIZE: number;

export declare function getDriverVersion(): number;
export declare function rgbaMirror(
  width: number, height: number, axis: number, source: any, target?: any): void;
export declare function bgraToYCrCb420(
  target: any, source: any, width: number, height: number): void;

export declare namespace Math {
  export function abs<T extends number|bigint>(x: T): T;
  export function acos<T extends number|bigint>(x: T): T;
  export function asin<T extends number|bigint>(x: T): T;
  export function atan<T extends number|bigint>(x: T): T;
  export function atan2<T extends number|bigint>(y: T, x: T): T;
  export function ceil<T extends number|bigint>(x: T): T;
  export function cos<T extends number|bigint>(x: T): T;
  export function exp<T extends number|bigint>(x: T): T;
  export function floor<T extends number|bigint>(x: T): T;
  export function log<T extends number|bigint>(x: T): T;
  export function max<T extends number|bigint>(...values: T[]): T;
  export function min<T extends number|bigint>(...values: T[]): T;
  export function pow<T extends number|bigint>(x: T, y: T): T;
  export function round<T extends number|bigint>(x: T): T;
  export function sin<T extends number|bigint>(x: T): T;
  export function sqrt<T extends number|bigint>(x: T): T;
  export function tan<T extends number|bigint>(x: T): T;
}

export declare namespace driver {
  /** @ignore */
  export enum PointerAttributes {
    CONTEXT,
    MEMORY_TYPE,
    DEVICE_POINTER,
    HOST_POINTER,
    // P2P_TOKENS,
    SYNC_MEMOPS,
    BUFFER_ID,
    IS_MANAGED,
    DEVICE_ORDINAL,
  }

  export function cuPointerGetAttribute(mem: ArrayBuffer|ArrayBufferView|MemoryData,
                                        attr: PointerAttributes): any;
}

export declare namespace runtime {
  /**
   * Flags to register a graphics resource
   * @ignore
   */
  export enum GraphicsRegisterFlags {
    NONE,
    READ_ONLY,
    WRITE_DISCARD,
  }

  export function cudaMemGetInfo(): {free: number, total: number};
  export function cudaMemset(
    target: MemoryData, value: number, count: number, stream?: number): void;
  export function cudaMemcpy(
    target: MemoryData, source: MemoryData, count: number, stream?: number): void;

  export function cudaGLGetDevices(list: 0|1|2): number[];
  export function cudaGraphicsGLRegisterBuffer(glBuffer: number, flags: number): number;
  export function cudaGraphicsGLRegisterImage(
    glImage: number, target: number, flags: number): number;
  export function cudaGraphicsUnregisterResource(resource: number): void;
  export function cudaGraphicsMapResources(resources: number[]): void;
  export function cudaGraphicsUnmapResources(resources: number[]): void;
  export function cudaGraphicsResourceGetMappedArray(resource: number): CUDAArray;
  export function cudaGraphicsResourceGetMappedPointer(resource: number): MappedGLMemory;
}

/**
 * CUDAArray channel format kind
 * @ignore
 */
export declare enum ChannelFormatKind {
  /** Signed channel format */
  SIGNED,
  /** Unsigned channel format */
  UNSIGNED,
  /** Float channel format */
  FLOAT,
  /** No channel format */
  NONE,
}

/** @ignore */
export declare class CUDAArray {
  private constructor(ptr: number,
                      extent: {width: number, height: number, depth: number},
                      channelFormatDesc: {
                        x: number,
                        y: number,
                        z: number,
                        w: number,
                        f: ChannelFormatKind,
                      },
                      flags: number,
                      type: 0|1|2)
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
  readonly channelFormatKind: ChannelFormatKind;
}

/**
 * @summary The flags for the {@link Device}'s primary context.
 *
 * @description The three LSBs of the `flags` parameter can be used to control
 * how the OS thread, which owns the CUDA context at the time of an API call,
 * interacts with the OS scheduler when waiting for results from the GPU.
 * Only one of the scheduling flags can be set when creating a context.
 */
export declare enum DeviceFlags {
  /**
   *  Uses a heuristic based on the number of active CUDA contexts in the
   *  process `C` and the number of logical processors in the system `P`.
   *  If `C` > `P`, then CUDA will yield to other OS threads when waiting
   *  for the GPU (`DeviceFlag.scheduleYield`), otherwise CUDA will not
   *  yield while waiting for results and actively spin on the processor
   *  (`DeviceFlag.scheduleSpin`).
   *  <br/>
   *  Additionally, on Tegra devices, `DeviceFlag.scheduleAuto` uses a
   *  heuristic based on the power profile of the platform and may choose
   *  `DeviceFlag.scheduleBlockingSync` for low-powered devices.
   */
  scheduleAuto,
  /**
   *  Instruct CUDA to actively spin when waiting for results from the GPU.
   *  This can decrease latency when waiting for the GPU, but may lower the
   *  performance of CPU threads if they are performing work in parallel
   *  with the CUDA thread.
   */
  scheduleSpin,
  /**
   *  Instruct CUDA to yield its thread when waiting for results from the
   *  GPU. This can increase latency when waiting for the GPU, but can
   *  increase the performance of CPU threads performing work in parallel
   *  with the GPU.
   */
  scheduleYield,
  /**
   *  Instruct CUDA to block the CPU thread on a synchronization primitive
   *  when waiting for the GPU to finish work.
   */
  scheduleBlockingSync,
  /**
   * @ignore
   */
  mapHost,
  /**
   *  Instruct CUDA to not reduce local memory after resizing local memory
   *  for a kernel. This can prevent thrashing by local memory allocations
   *  when launching many kernels with high local memory usage at the cost
   *  of potentially increased memory usage.
   */
  lmemResizeToMax,
}

export declare class Device {
  /**
   * The number of compute-capable CUDA devices.
   */
  static readonly numDevices: number;
  /**
   * The id of this thread's active CUDA device.
   */
  static readonly activeDeviceId: number;

  constructor(deviceId?: number, flags?: DeviceFlags);

  /**
   * The CUDA device identifer
   */
  readonly id: number;

  /**
   * The CUDA device PCI bus string id
   */
  readonly pciBusName: string;

  /**
   * @summary Destroy all allocations and reset all state on the current
   * device in the current process.
   *
   * @description
   * Explicitly destroys and cleans up all resources associated with the
   * current device in the current process. Any subsequent API call to
   * this device will reinitialize the device.
   * <br/><br/>
   * Note that this function will reset the device immediately. It is the
   * caller's responsibility to ensure that the device is not being accessed
   * by any other host threads from the process when this function is called.
   */
  reset(): this;

  /**
   * @summary Set this device to be used for GPU executions.
   *
   * @description
   * Sets this device as the current device for the calling host thread.
   * <br/><br/>
   * Any device memory subsequently allocated from this host thread
   * will be physically resident on this device. Any host memory allocated
   * from this host thread will have its lifetime associated with this
   * device. Any streams or events created from this host thread will
   * be associated with this device. Any kernels launched from this host
   * thread will be executed on this device.
   * <br/><br/>
   * This call may be made from any host thread, to any device, and at
   * any time. This function will do no synchronization with the previous
   * or new device, and should be considered a very low overhead call.
   */
  activate(): this;

  /**
   * @summary Get the {@link DeviceFlag_ device flags} used to initialize this device.
   */
  getFlags(): DeviceFlags;

  /**
   * @summary Set the {@link DeviceFlag device flags} for the device's primary context.
   *
   * @param {DeviceFlags} newFlags The new flags for the device's primary context.
   */
  setFlags(newFlags: DeviceFlags): void;

  /**
   * @summary An object with information about the device.
   */
  getProperties(): DeviceProperties;

  /**
   * @summary Wait for this compute device to finish.
   *
   * @description
   * Blocks execution of further device calls until the device has completed
   * all preceding requested tasks.
   *
   * @throws an error if one of the preceding tasks has failed. If the
   * `cudaDeviceScheduleBlockingSync` flag was set for this device, the
   * host thread will block until the device has finished its work.
   */
  synchronize(): this;

  /**
   * @summary Ensures this device is active, then executes the supplied `work` function.
   * <br/><br/>
   * If the current device was not already the active device, restores the active device after the
   * `work` function has completed.
   * @param work A function to execute
   */
  callInContext(work: () => any): this;

  /**
   * @summary Queries if a device may directly access a peer device's memory.
   * <br/><br/>
   * If direct access of `peerDevice` from this device is possible, then
   * access may be enabled on two specific devices by calling
   * {@link enablePeerAccess}.
   *
   * @returns `true` if this Device's contexts are capable of directly
   * accessing memory from contexts on `peerDevice`, otherwise `false`.
   */
  canAccessPeerDevice(peerDevice: Device): boolean;

  /**
   * @summary Enables direct access to memory allocations in a peer device.
   */
  enablePeerAccess(peerDevice: Device): this;

  /**
   * @summary Disables direct access to memory allocations in a peer device and unregisters any
   * registered allocations.
   */
  disablePeerAccess(peerDevice: Device): this;
}

export declare interface DeviceProperties {
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
  /**
   * Device can concurrently copy memory and execute a kernel. Deprecated. Use instead
   * asyncEngineCount.
   */
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
  /**
   * Ratio of single precision performance (in floating-point operations per second) to double
   * precision performance
   */
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
  /**
   * Device can participate in cooperative kernels launched via
   * ::cudaLaunchCooperativeKernelMultiDevice
   */
  cooperativeMultiDeviceLaunch: number;
  /** Per device maximum shared memory per block usable by special opt in */
  sharedMemPerBlockOptin: number;
  /** Device accesses pageable memory via the host's page tables */
  pageableMemoryAccessUsesHostPageTables: number;
  /** Host can directly access managed memory on the device without migration. */
  directManagedMemAccessFromHost: number;
}

/** @ignore */
export declare class Memory extends ArrayBuffer {
  /** @ignore */
  readonly ptr: number;
  /**
   * @summary The {@link Device device} this Memory instance is associated with.
   */
  readonly device: number;
  /**
   * @summary Copies and returns a region of Memory.
   */
  slice(start?: number, end?: number): Memory;
}

/**
 * @summary An owning wrapper around a device memory allocation.
 */
export declare class DeviceMemory extends Memory {
  constructor(byteLength?: number);
  /** @ignore */
  readonly[Symbol.toStringTag]: 'DeviceMemory';
  /**
   * @summary Copies and returns a region of DeviceMemory.
   */
  slice(start?: number, end?: number): DeviceMemory;
}

/**
 * @brief An owning wrapper around a pinned host memory allocation.
 */
export declare class PinnedMemory extends Memory {
  constructor(byteLength?: number);
  /** @ignore */
  readonly[Symbol.toStringTag]: 'PinnedMemory';
  /**
   * @summary Copies and returns a region of PinnedMemory.
   */
  slice(start?: number, end?: number): PinnedMemory;
}

/**
 * @brief An owning wrapper around a CUDA-managed, unified memory allocation.
 */
export declare class ManagedMemory extends Memory {
  constructor(byteLength?: number);
  /** @ignore */
  readonly[Symbol.toStringTag]: 'ManagedMemory';
  /**
   * @summary Copies and returns a region of ManagedMemory.
   */
  slice(start?: number, end?: number): ManagedMemory;
}

/**
 * @summary An owning wrapper around a CUDA device memory allocation shared by another process.
 */
export declare class IpcMemory extends Memory {
  constructor(ipcHandle: Uint8Array);
  /** @ignore */
  readonly[Symbol.toStringTag]: 'IpcMemory';
  /**
   * @summary Copies a region of IpcMemory and returns as DeviceMemory.
   */
  slice(start?: number, end?: number): DeviceMemory;
  /**
   * @summary Close the underlying IPC memory handle, allowing the exporting process to free the
   * exported {@link DeviceMemory}.
   */
  close(): void;
}

/**
 * @summary A container for managing the lifetime of a {@link DeviceMemory} allocation exported for
 * reading and/or writing by other processes with access to the allocation's associated {@link
 * Device}.
 */
export declare class IpcHandle {
  constructor(deviceMemory: DeviceMemory);
  /** @ignore */
  readonly[Symbol.toStringTag]: 'IpcHandle';
  /**
   * @summary The exported {@link DeviceMemory}
   */
  readonly buffer: DeviceMemory;
  /**
   * @summary The device ordinal associated with the exported {@link DeviceMemory}
   */
  readonly device: number;
  /**
   * @summary The CUDA IPC handle to be used to access the exported {@link DeviceMemory} from
   * another process.
   */
  readonly handle: Uint8Array;
}

/**
 * @summary A class representing a region of memory originally created and owned by an OpenGL
 * context, but has been mapped into the CUDA address space for reading and/or writing.
 */
export declare class MappedGLMemory extends Memory {
  constructor(resource: number);
  /** @ignore */
  readonly[Symbol.toStringTag]: 'MappedGLMemory';
  /**
   * @summary Copies a region of MappedGLMemory and returns a DeviceMemory.
   */
  slice(start?: number, end?: number): DeviceMemory;
}
