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

import { TypedArray, BigIntArray } from './interfaces';
import { DeviceFlag, DeviceProperties } from './device';

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
    if (CUDA) return CUDA.init() as CUDA;
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

    readonly Device: DeviceConstructor;
    readonly DeviceMemory: DeviceMemoryConstructor;
    readonly PinnedMemory: PinnedMemoryConstructor;
    readonly ManagedMemory: ManagedMemoryConstructor;
    readonly IpcMemory: IpcMemoryConstructor;
    readonly IpcHandle: IpcHandleConstructor;

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

interface DeviceConstructor {
    /**
     * The number of compute-capable CUDA devices.
     */
    readonly numDevices: number;
    /**
     * The id of this thread's active CUDA device.
     */
    readonly activeDeviceId: number;
    readonly prototype: Device;
    new(deviceId?: number, flags?: DeviceFlag): Device;
}

interface Device {

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
     * device in the current process. Resets the device with the specified
     * {@link DeviceFlag} device flags.
     * 
     * @description
     * Explicitly destroys and cleans up all resources associated with the
     * current device in the current process. Any subsequent API call to
     * this device will reinitialize the device.
     *
     * Note that this function will reset the device immediately. It is the
     * caller's responsibility to ensure that the device is not being accessed
     * by any other host threads from the process when this function is called.
     * 
     * @param {DeviceFlag} flags The flags for the device's primary
     * context.
     * <br/>
     */
    reset(flags?: DeviceFlag): this;

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
     * @summary Queries the {@link DeviceFlag} flags used to initialize this device.
     */
    getFlags(): DeviceFlag;

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
    callInContext(work: Function): this;

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

/** @ignore */
export type MemoryData = TypedArray | BigIntArray | ArrayBufferView | ArrayBufferLike | Memory;

export interface Memory extends ArrayBuffer {
    readonly ptr: number;
    readonly device: number;
    slice(start?: number, end?: number): Memory;
}

export interface MemoryConstructor {
    new(byteLength?: number): Memory;
}

interface DeviceMemoryConstructor extends MemoryConstructor {
    readonly prototype: DeviceMemory;
    new(byteLength?: number): Memory;
}

interface DeviceMemory {
    slice(start?: number, end?: number): DeviceMemory;
}

interface PinnedMemoryConstructor extends MemoryConstructor {
    readonly prototype: PinnedMemory;
    new(byteLength?: number): Memory;
}

interface PinnedMemory {
    slice(start?: number, end?: number): PinnedMemory;
}

interface ManagedMemoryConstructor extends MemoryConstructor {
    readonly prototype: ManagedMemory;
    new(byteLength?: number): Memory;
}

interface ManagedMemory {
    slice(start?: number, end?: number): ManagedMemory;
}

interface IpcMemoryConstructor {
    readonly prototype: IpcMemory;
    new(ipcHandle: Uint8Array): Memory;
}

interface IpcMemory {
    slice(start?: number, end?: number): DeviceMemory;
}

interface IpcHandleConstructor {
    readonly prototype: IpcHandle;
    new(deviceMemory: DeviceMemory): IpcHandle;
}

interface IpcHandle {
    readonly device: number;
    readonly handle: Uint8Array;
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
