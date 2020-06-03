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

import * as CUDA from './cuda';
import {
    CUdevice,
    CUDADeviceFlag,
    CUDADeviceProperties
} from './cuda';

export class CUDADevice {

    /**
     * @summary Creates a new {@link CUDADevice} at the given ordinal with the specified {@link CUDADeviceFlag} flags.
     * @param {CUDADeviceFlag} flags The flags for the device's primary context.
     * @see {@link CUDADevice.prototype.reset}
     */
    public static new(deviceIndex: number, flags: CUDADeviceFlag = CUDADeviceFlag.scheduleAuto) {
        return new CUDADevice(deviceIndex, flags);
    }

    private constructor(deviceIndex: number, flags: CUDADeviceFlag = CUDADeviceFlag.scheduleAuto) {
        this.id = CUDA.device.getByIndex(deviceIndex);
        this.reset(flags).activate();
    }

    public readonly id: CUdevice;

    // @ts-ignore
    protected _properties: CUDADeviceProperties;
    /**
     * @summary An object with information about the device.
     */
    public get properties() {
        return this._properties || (this._properties = CUDA.device.getProperties(this.id));
    }

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
    public activate() {
        if (CUDA.device.get() !== this.id) {
            CUDA.device.set(this.id);
        }
        return this;
    }

    /**
     * @summary Destroy all allocations and reset all state on the current
     * device in the current process. Resets the device with the specified
     * {@link CUDADeviceFlag} device flags.
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
     * @param {CUDADeviceFlag} flags The flags for the device's primary
     * context.
     * <br/>
     * The three LSBs of the `flags` parameter can be used to control how
     * the OS thread, which owns the CUDA context at the time of an API call,
     * interacts with the OS scheduler when waiting for results from the GPU.
     * Only one of the scheduling flags can be set when creating a context.
     * 
     * * `CUDADeviceFlag.scheduleAuto`:
     *  Uses a heuristic based on the number of active CUDA contexts in the
     *  process `C` and the number of logical processors in the system `P`.
     *  If `C` > `P`, then CUDA will yield to other OS threads when waiting
     *  for the GPU (`CUDADeviceFlag.scheduleYield`), otherwise CUDA will not
     *  yield while waiting for results and actively spin on the processor
     *  (`CUDADeviceFlag.scheduleSpin`).
     *  <br/>
     *  Additionally, on Tegra devices, `CUDADeviceFlag.scheduleAuto` uses a
     *  heuristic based on the power profile of the platform and may choose
     *  `CUDADeviceFlag.scheduleBlockingSync` for low-powered devices.
     * * `CUDADeviceFlag.scheduleSpin`:
     *  Instruct CUDA to actively spin when waiting for results from the GPU.
     *  This can decrease latency when waiting for the GPU, but may lower the
     *  performance of CPU threads if they are performing work in parallel
     *  with the CUDA thread.
     * * `CUDADeviceFlag.scheduleYield`:
     *  Instruct CUDA to yield its thread when waiting for results from the
     *  GPU. This can increase latency when waiting for the GPU, but can
     *  increase the performance of CPU threads performing work in parallel
     *  with the GPU.
     * * `CUDADeviceFlag.scheduleBlockingSync`:
     *  Instruct CUDA to block the CPU thread on a synchronization primitive
     *  when waiting for the GPU to finish work.
     * * `CUDADeviceFlag.lmemResizeToMax`:
     *  Instruct CUDA to not reduce local memory after resizing local memory
     *  for a kernel. This can prevent thrashing by local memory allocations
     *  when launching many kernels with high local memory usage at the cost
     *  of potentially increased memory usage.
     */
    public reset(flags: CUDADeviceFlag = CUDADeviceFlag.scheduleAuto) {
        return this.callInContext(() => {
            CUDA.device.reset();
            CUDA.device.setFlags(flags);
            CUDA.device.synchronize();
            return this;
        });
    }

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
    public synchronize() { this.callInContext(() => CUDA.device.synchronize()); return this; }

    /**
     * @summary Queries the {@link CUDADeviceFlag} flags used to initialize this device.
     */
    public getFlags() { return this.callInContext(() => CUDA.device.getFlags() - CUDADeviceFlag.mapHost); }

    /**
     * @summary Ensures this device is active, then executes the supplied @p `work` function. Restores the active
     * device after executing the function (if the current device was not already the active device).
     * @param work A function to execute
     */
    public callInContext<T>(work: (...args: any) => T): T {
        let result: T;
        const current = CUDA.device.get();
        try {
            current !== this.id && CUDA.device.set(this.id);
            result = work();
        } finally {
            current !== this.id && CUDA.device.set(current);
        }
        return result;
    }

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
    public canAccessPeerDevice(peerDevice: CUDADevice) { return CUDA.device.canAccessPeer(this.id, peerDevice.id); }
    /**
     * @summary Enables direct access to memory allocations in a peer device.
     */
    public enablePeerAccess(peerDevice: CUDADevice) { this.callInContext(() => CUDA.device.enablePeerAccess(peerDevice.id)); return this; }
    /**
     * @summary Disables direct access to memory allocations in a peer device and unregisters any registered allocations.
     */
    public disablePeerAccess(peerDevice: CUDADevice) { this.callInContext(() => CUDA.device.disablePeerAccess(peerDevice.id)); return this; }

    public get [Symbol.toStringTag]() { return 'CUDADevice'; }
    public [Symbol.for('nodejs.util.inspect.custom')]() { return this.toString(); }
    public toString() {
        return `${this[Symbol.toStringTag]} ${JSON.stringify({
            'id': this.id,
            'name': this.properties.name,
            'compute capability': [
                this.properties.major,
                this.properties.minor
            ]
        })}`;
    }
}
