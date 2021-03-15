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

import CUDA from './addon';

/** @ignore */
export interface Memory extends ArrayBuffer {
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

/** @ignore */
export interface DeviceMemoryConstructor {
  readonly prototype: DeviceMemory;
  new(byteLength?: number): DeviceMemory;
}

/**
 * @summary An owning wrapper around a device memory allocation.
 */
export interface DeviceMemory extends Memory {
  /** @ignore */
  readonly[Symbol.toStringTag]: 'DeviceMemory';
  /**
   * @summary Copies and returns a region of DeviceMemory.
   */
  slice(start?: number, end?: number): DeviceMemory;
}

/** @ignore */
// eslint-disable-next-line @typescript-eslint/no-redeclare
export const DeviceMemory: DeviceMemoryConstructor = CUDA.DeviceMemory;

/** @ignore */
export interface PinnedMemoryConstructor {
  readonly prototype: PinnedMemory;
  new(byteLength?: number): PinnedMemory;
}

/**
 * @brief An owning wrapper around a pinned host memory allocation.
 */
export interface PinnedMemory extends Memory {
  /** @ignore */
  readonly[Symbol.toStringTag]: 'PinnedMemory';
  /**
   * @summary Copies and returns a region of PinnedMemory.
   */
  slice(start?: number, end?: number): PinnedMemory;
}

/** @ignore */
// eslint-disable-next-line @typescript-eslint/no-redeclare
export const PinnedMemory: PinnedMemoryConstructor = CUDA.PinnedMemory;

/** @ignore */
export interface ManagedMemoryConstructor {
  readonly prototype: ManagedMemory;
  new(byteLength?: number): ManagedMemory;
}

/**
 * @brief An owning wrapper around a CUDA-managed, unified memory allocation.
 */
export interface ManagedMemory extends Memory {
  /** @ignore */
  readonly[Symbol.toStringTag]: 'ManagedMemory';
  /**
   * @summary Copies and returns a region of ManagedMemory.
   */
  slice(start?: number, end?: number): ManagedMemory;
}

/** @ignore */
// eslint-disable-next-line @typescript-eslint/no-redeclare
export const ManagedMemory: ManagedMemoryConstructor = CUDA.ManagedMemory;

/** @ignore */
export interface IpcMemoryConstructor {
  readonly prototype: IpcMemory;
  new(ipcHandle: Uint8Array): IpcMemory;
}

/**
 * @summary An owning wrapper around a CUDA device memory allocation shared by another process.
 */
export interface IpcMemory extends Memory {
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

/** @ignore */
// eslint-disable-next-line @typescript-eslint/no-redeclare
export const IpcMemory: IpcMemoryConstructor = CUDA.IpcMemory;

/** @ignore */
export interface IpcHandleConstructor {
  readonly prototype: CUDAIpcHandle;
  new(deviceMemory: DeviceMemory): CUDAIpcHandle;
}

interface CUDAIpcHandle {
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
  /**
   * @summary Close the underlying IPC memory handle, allowing this process to free the
   * exported {@link DeviceMemory}.
   */
  close(): void;
}

/**
 * @summary A container for managing the lifetime of a {@link DeviceMemory} allocation exported for
 * reading and/or writing by other processes with access to the allocation's associated {@link
 * Device}.
 */
export class IpcHandle extends(<IpcHandleConstructor>CUDA.IpcHandle) {
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
   * @returns An object with the device ordinal, byte offset (if applicable) into the
   *   exported {@link DeviceMemory}, and the 64-bit IPC handle (as a JavaScript Array of octets).
   */
  public toJSON() {
    return {
      device: this.device,
      byteOffset: this.byteOffset,
      handle: [...this.handle],
    };
  }
}

/** @ignore */
export interface MappedGLMemoryConstructor {
  readonly prototype: MappedGLMemory;
  new(resource: number): MappedGLMemory;
}

/**
 * @summary A class representing a region of memory originally created and owned by an OpenGL
 * context, but has been mapped into the CUDA address space for reading and/or writing.
 */
export interface MappedGLMemory extends Memory {
  /** @ignore */
  readonly[Symbol.toStringTag]: 'MappedGLMemory';
  /**
   * @summary Copies a region of MappedGLMemory and returns a DeviceMemory.
   */
  slice(start?: number, end?: number): DeviceMemory;
}

/** @ignore */
// eslint-disable-next-line @typescript-eslint/no-redeclare
export const MappedGLMemory: MappedGLMemoryConstructor = CUDA.MappedGLMemory;
