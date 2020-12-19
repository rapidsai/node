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

import CUDA from './addon';

export class Device extends (<CUDADeviceConstructor>CUDA.Device) {
  /**
   * The number of compute-capable CUDA devices.
   */
  static readonly numDevices: number;
  /**
   * The id of this thread's active CUDA device.
   */
  static readonly activeDeviceId: number;

  public get name(): string {
    return this.getProperties().name;
  }
  public get pciBusId(): number {
    return this.getProperties().pciBusID;
  }

  // @ts-ignore
  protected _properties: DeviceProperties;

  /**
   * @summary An object with information about the device.
   */
  public getProperties() {
    return this._properties || (this._properties = super.getProperties());
  }

  public get [Symbol.toStringTag]() {
    return 'CUDADevice';
  }
  public [Symbol.for('nodejs.util.inspect.custom')]() {
    return this.toString();
  }
  public toString() {
    const { name, major, minor } = this.getProperties();
    return `${this[Symbol.toStringTag]} ${JSON.stringify({
      'id': this.id,
      'name': name,
      'compute_capability': [major, minor],
    })}`;
  }
}

export const devices = new Proxy<DeviceList>(
  {
    length: Device.numDevices,
    *[Symbol.iterator]() {
      for (let i = -1, n = this.length; ++i < n; yield this[i]);
    },
  },
  {
    isExtensible() {
      return false;
    },
    set() {
      throw new Error('Invalid operation');
    },
    defineProperty() {
      throw new Error('Invalid operation');
    },
    deleteProperty() {
      throw new Error('Invalid operation');
    },
    has(target, key) {
      return typeof key !== 'number' ? key in target : key > -1 && key < Device.numDevices;
    },
    get(target, key) {
      const idx = typeof key !== 'symbol' ? +(key as any) : NaN;
      if (idx == idx && idx > -1 && idx < Device.numDevices) {
        return target[idx] ? target[idx] : (target[idx] = new Device(idx));
      }
      return target[key as any];
    },
  },
);

interface DeviceList extends Iterable<Device> {
  length: number;
  [key: number]: Device;
}

interface CUDADeviceConstructor {
  /**
   * The number of compute-capable CUDA devices.
   */
  readonly numDevices: number;
  /**
   * The id of this thread's active CUDA device.
   */
  readonly activeDeviceId: number;
  readonly prototype: CUDADevice;
  new (deviceId?: number, flags?: DeviceFlags): CUDADevice;
}

interface CUDADevice {
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
   *
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
   *
   * Any device memory subsequently allocated from this host thread
   * will be physically resident on this device. Any host memory allocated
   * from this host thread will have its lifetime associated with this
   * device. Any streams or events created from this host thread will
   * be associated with this device. Any kernels launched from this host
   * thread will be executed on this device.
   *
   * This call may be made from any host thread, to any device, and at
   * any time. This function will do no synchronization with the previous
   * or new device, and should be considered a very low overhead call.
   */
  activate(): this;

  /**
   * @summary Get the {@link DeviceFlag} flags used to initialize this device.
   */
  getFlags(): DeviceFlags;

  /**
   * @summary Set the {@link DeviceFlag} flags for the device's primary context.
   *
   * @param {DeviceFlag} newFlags The new flags for the device's primary context.
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
   * @summary Ensures this device is active, then executes the supplied @p `work` function. Restores the active
   * device after executing the function (if the current device was not already the active device).
   * @param work A function to execute
   */
  callInContext(work: () => any): this;

  /**
   * @summary Queries if a device may directly access a peer device's memory.
   *
   * If direct access of `peerDevice` from this device is possible, then
   * access may be enabled on two specific contexts by calling
   * {@link CUDAContext.prototype.enablePeerAccess}.
   *
   * @returns `true` if this device's contexts are capable of directly
   * accessing memory from contexts on `peerDevice` , otherwise `false`.
   */
  canAccessPeerDevice(peerDevice: Device): boolean;

  /**
   * @summary Enables direct access to memory allocations in a peer device.
   */
  enablePeerAccess(peerDevice: Device): this;

  /**
   * @summary Disables direct access to memory allocations in a peer device and unregisters any registered allocations.
   */
  disablePeerAccess(peerDevice: Device): this;
}

/**
 * @summary The flags for the {@link Device}'s primary context.
 *
 * @description The three LSBs of the `flags` parameter can be used to control
 * how the OS thread, which owns the CUDA context at the time of an API call,
 * interacts with the OS scheduler when waiting for results from the GPU.
 * Only one of the scheduling flags can be set when creating a context.
 */
export enum DeviceFlags {
  /**
   *  Uses a heuristic based on the number of active CUDA contexts in the
   *  process `C` and the number of logical processors in the system `P`.
   *  If `C` > `P`, then CUDA will yield to other OS threads when waiting
   *  for the GPU (`DeviceFlags.scheduleYield`), otherwise CUDA will not
   *  yield while waiting for results and actively spin on the processor
   *  (`DeviceFlags.scheduleSpin`).
   *  <br/>
   *  Additionally, on Tegra devices, `DeviceFlags.scheduleAuto` uses a
   *  heuristic based on the power profile of the platform and may choose
   *  `DeviceFlags.scheduleBlockingSync` for low-powered devices.
   */
  scheduleAuto = CUDA.DeviceFlags.scheduleAuto,
  /**
   *  Instruct CUDA to actively spin when waiting for results from the GPU.
   *  This can decrease latency when waiting for the GPU, but may lower the
   *  performance of CPU threads if they are performing work in parallel
   *  with the CUDA thread.
   */
  scheduleSpin = CUDA.DeviceFlags.scheduleSpin,
  /**
   *  Instruct CUDA to yield its thread when waiting for results from the
   *  GPU. This can increase latency when waiting for the GPU, but can
   *  increase the performance of CPU threads performing work in parallel
   *  with the GPU.
   */
  scheduleYield = CUDA.DeviceFlags.scheduleYield,
  /**
   *  Instruct CUDA to block the CPU thread on a synchronization primitive
   *  when waiting for the GPU to finish work.
   */
  scheduleBlockingSync = CUDA.DeviceFlags.scheduleBlockingSync,
  /**
   * @ignore
   */
  mapHost = CUDA.DeviceFlags.mapHost,
  /**
   *  Instruct CUDA to not reduce local memory after resizing local memory
   *  for a kernel. This can prevent thrashing by local memory allocations
   *  when launching many kernels with high local memory usage at the cost
   *  of potentially increased memory usage.
   */
  lmemResizeToMax = CUDA.DeviceFlags.lmemResizeToMax,
}

export interface DeviceProperties {
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
