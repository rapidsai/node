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

export interface Memory extends ArrayBuffer {
  readonly ptr: number;
  readonly device: number;
  slice(start?: number, end?: number): Memory;
}

export interface DeviceMemoryConstructor {
  readonly prototype: DeviceMemory;
  new (byteLength?: number): DeviceMemory;
}

export interface DeviceMemory extends Memory {
  readonly [Symbol.toStringTag]: 'DeviceMemory';
  slice(start?: number, end?: number): DeviceMemory;
}

// eslint-disable-next-line @typescript-eslint/no-redeclare
export const DeviceMemory: DeviceMemoryConstructor = CUDA.DeviceMemory;

export interface PinnedMemoryConstructor {
  readonly prototype: PinnedMemory;
  new (byteLength?: number): PinnedMemory;
}

export interface PinnedMemory extends Memory {
  readonly [Symbol.toStringTag]: 'PinnedMemory';
  slice(start?: number, end?: number): PinnedMemory;
}

// eslint-disable-next-line @typescript-eslint/no-redeclare
export const PinnedMemory: PinnedMemoryConstructor = CUDA.PinnedMemory;

export interface ManagedMemoryConstructor {
  readonly prototype: ManagedMemory;
  new (byteLength?: number): ManagedMemory;
}

export interface ManagedMemory extends Memory {
  readonly [Symbol.toStringTag]: 'ManagedMemory';
  slice(start?: number, end?: number): ManagedMemory;
}

// eslint-disable-next-line @typescript-eslint/no-redeclare
export const ManagedMemory: ManagedMemoryConstructor = CUDA.ManagedMemory;

export interface IpcMemoryConstructor {
  readonly prototype: IpcMemory;
  new (ipcHandle: Uint8Array): IpcMemory;
}

export interface IpcMemory extends Memory {
  readonly [Symbol.toStringTag]: 'IpcMemory';
  slice(start?: number, end?: number): DeviceMemory;
}

// eslint-disable-next-line @typescript-eslint/no-redeclare
export const IpcMemory: IpcMemoryConstructor = CUDA.IpcMemory;

export interface IpcHandleConstructor {
  readonly prototype: CUDAIpcHandle;
  new (deviceMemory: DeviceMemory): CUDAIpcHandle;
}

interface CUDAIpcHandle {
  readonly [Symbol.toStringTag]: 'IpcHandle';
  readonly buffer: DeviceMemory;
  readonly device: number;
  readonly handle: Uint8Array;
  close(): void;
}

export class IpcHandle extends (<IpcHandleConstructor>CUDA.IpcHandle) {
  constructor(deviceMemory: DeviceMemory, byteOffset = 0) {
    super(deviceMemory);
    this.byteOffset = byteOffset;
  }
  public readonly byteOffset: number;
  public [Symbol.for('nodejs.util.inspect.custom')]() {
    return `${this[Symbol.toStringTag]} ${this.toString()}`;
  }
  public toString() {
    return JSON.stringify(this.toJSON());
  }
  public toJSON() {
    return {
      device: this.device,
      byteOffset: this.byteOffset,
      handle: [...this.handle],
    };
  }
}

export interface MappedGLMemoryConstructor {
  readonly prototype: MappedGLMemory;
  new (resource: number): MappedGLMemory;
}

export interface MappedGLMemory extends Memory {
  readonly [Symbol.toStringTag]: 'ManagedMemory';
  slice(start?: number, end?: number): MappedGLMemory;
}

// eslint-disable-next-line @typescript-eslint/no-redeclare
export const MappedGLMemory: MappedGLMemoryConstructor = CUDA.MappedGLMemory;
