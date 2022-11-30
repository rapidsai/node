// Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

import {
  DeviceMemory,
  IpcHandle as CUDAIpcHandle,
  IpcMemory,
  ManagedMemory,
  MappedGLMemory,
  Memory,
  PinnedMemory,
} from './addon';

export {DeviceMemory, IpcMemory, ManagedMemory, MappedGLMemory, Memory, PinnedMemory};

/**
 * @summary A container for managing the lifetime of a {@link DeviceMemory} allocation exported for
 * reading and/or writing by other processes with access to the allocation's associated {@link
 * Device}.
 */
export class IpcHandle extends CUDAIpcHandle {
  constructor(deviceMemory: DeviceMemory, byteOffset = 0) {
    super(deviceMemory);
    this.byteOffset = byteOffset;
  }

  /**
   * @summary The byte offset (if applicable) into the exported {@link DeviceMemory}
   */
  public readonly byteOffset: number;

  /** @ignore */
  public[Symbol.for('nodejs.util.inspect.custom')]() {
    return `${this[Symbol.toStringTag]} ${this.toString()}`;
  }

  /**
   * @summary JSON-stringified details describing the exported {@link DeviceMemory} and CUDA IPC
   * handle.
   * @returns The result of calling `JSON.stringify(this.toJSON())`
   */
  public toString() { return JSON.stringify(this.toJSON()); }

  /**
   * @summary An object describing the exported {@link DeviceMemory} and CUDA IPC handle.
   * @returns An object with the device ordinal, the 64-bit IPC handle (as a JavaScript Array of
   *   octets), byte offset (if applicable) into the exported {@link DeviceMemory}, byte length of
   *   the IPC segment.
   */
  public toJSON() {
    return {
      device: this.device,
      handle: [...this.handle],
      byteOffset: this.byteOffset,
      byteLength: this.buffer.byteLength - this.byteOffset,
    };
  }
}
