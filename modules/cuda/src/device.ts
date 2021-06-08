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

import {Device as CUDADevice, DeviceFlags, DeviceProperties} from './addon';

export {DeviceFlags};

export class Device extends CUDADevice {
  /**
   * The number of compute-capable CUDA devices.
   */
  public static get numDevices() { return CUDADevice.numDevices; }
  /**
   * The id of this thread's active CUDA device.
   */
  public static get activeDeviceId() { return CUDADevice.activeDeviceId; }

  /**
   * The human-readable name of this CUDA Device
   */
  public get name(): string { return this.getProperties().name; }
  /**
   * The PCI Bus identifier of this CUDA Device
   */
  public get pciBusId(): number { return this.getProperties().pciBusID; }

  // @ts-ignore
  protected _properties: DeviceProperties;

  /**
   * @summary An object with information about the device.
   */
  public getProperties() { return this._properties || (this._properties = super.getProperties()); }

  /** @ignore */
  public get[Symbol.toStringTag]() { return 'CUDADevice'; }

  /** @ignore */
  public[Symbol.for('nodejs.util.inspect.custom')]() { return this.toString(); }

  /** @ignore */
  public toString() {
    const {name, major, minor} = this.getProperties();
    return `${this[Symbol.toStringTag]} ${
      JSON.stringify({'id': this.id, 'name': name, 'compute_capability': [major, minor]})}`;
  }
}

interface DeviceList extends Iterable<Device> {
  length: number;
  [key: number]: Device;
}

/**
 * @summary A lazily-evaluated list of available CUDA devices.
 * <br/><br/>
 * This list has a `length` property, and each available active Device can be accessed by device
 * ordinal (via Array-style subscript-accesses).
 * <br/><br/>
 * This list implements the Iterable<Device> protocol, meaning it can be enumerated in a `for..of`
 * loop, or with the `[...]` iterable expansion syntax.
 *
 * @note While this list may seem like an Array, it is a JavaScript Proxy that only creates and
 * returns a Device instance for a given device ordinal the first time it's accessed.
 * @note Enumerating the `devices` list (i.e. `[...devices]`) will create and cache Device instances
 * for all CUDA devices available to the current process.
 *
 * @example
 * ```typescript
 * import {Device, devices} from '@nvidia/cuda';
 *
 * console.log(`Number of devices: ${devices.length}`);
 *
 * // CUDA Device 0 is automatically activated by default
 * console.log(`Active device id: ${Device.activeDeviceId}`); // 0
 *
 * // Access (and create) Devices 0,1
 * const [device0, device1] = devices;
 *
 * console.log(device0);
 * // > CUDADevice {"id":0,"name":"Quadro RTX 8000","compute_capability":[7,5]}
 *
 * console.log(device0.pciBusName);
 * // > '0000:15:00.0'
 *
 * console.log(device0.canAccessPeerDevice(device1));
 * // > true
 *
 * console.log(device0.getProperties());
 * // > {
 * // >   name: 'Quadro RTX 8000',
 * // >   totalGlobalMem: 50944540672,
 * // >   sharedMemPerBlock: 49152,
 * // >   regsPerBlock: 65536,
 * // >   warpSize: 32,
 * // >   memPitch: 2147483647,
 * // >   maxThreadsPerBlock: 1024,
 * // >   ...
 * // > }
 *
 * // Device 0 remains the active device until `device1` is made active
 * console.log(`Active device id: ${Device.activeDeviceId}`);
 * // > 0
 *
 * device1.activate();
 * console.log(`Active device id: ${Device.activeDeviceId}`);
 * // > 1
 *
 * // Set Device 0 to the active device again
 * device0.activate();
 * console.log(`Active device id: ${Device.activeDeviceId}`);
 * // > 0
 * ```
 */
export const devices = new Proxy<DeviceList>(
  {
    length: Device.numDevices,
    * [Symbol.iterator]() {
        for (let i = -1, n = this.length; ++i < n;) { yield this[i]; }
      }
  },
  {
    isExtensible() { return false;},
    set() { throw new Error('Invalid operation');},
    defineProperty() { throw new Error('Invalid operation');},
    deleteProperty() { throw new Error('Invalid operation');},
    has(target, key) {  //
      const idx = typeof key !== 'symbol' ? +(key as any) : NaN;
      return (idx !== idx) ? key in target : idx > -1 && idx < Device.numDevices;
    },
    get(target, key) {
      const idx = typeof key !== 'symbol' ? +(key as any) : NaN;
      if (idx == idx && idx > -1 && idx < Device.numDevices) {
        return target[idx] ? target[idx] : (target[idx] = new Device(idx));
      }
      return target[key as any];
    },
  });
