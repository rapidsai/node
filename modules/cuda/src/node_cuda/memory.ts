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

import CUDA from '../addon';

import { clampSliceArgs as clamp } from '../util';
import { TypedArray, BigIntArray } from '../interfaces';

/** @ignore */
export type MemoryData = TypedArray | BigIntArray | ArrayBufferView | ArrayBufferLike | Memory;

interface MemoryOperations {
    fill(target: MemoryData, value: number, count: number, stream?: number): number;
    copy(target: MemoryData, source: MemoryData, count: number, stream?: number): number;
}

export interface MemoryConstructor extends MemoryOperations {
    new(byteLength?: number): Memory;
}

export interface Memory extends ArrayBuffer {
    readonly ptr: number;
    readonly device: number;
    readonly constructor: MemoryOperations;
    slice(start?: number, end?: number): Memory;
}

interface IpcMemoryConstructor extends MemoryOperations {
    readonly prototype: Memory;
    new(ipcMemHandle: Uint8Array): Memory;
}

export class IpcMemory extends (<IpcMemoryConstructor> CUDA.IpcMemory) {
    slice(start?: number, end?: number): DeviceMemory {
        [start, end] = clamp(this.byteLength, start, end);
        return super.slice(start, end - start) as DeviceMemory;
    }
    public get [Symbol.toStringTag]() { return 'IpcMemory'; }
}

interface DeviceMemoryConstructor extends MemoryConstructor {
    readonly prototype: Memory;
    new(byteLength?: number): Memory;
}

export class DeviceMemory extends (<DeviceMemoryConstructor> CUDA.DeviceMemory) {
    slice(start?: number, end?: number): DeviceMemory {
        [start, end] = clamp(this.byteLength, start, end);
        return super.slice(start, end - start) as DeviceMemory;
    }
    public get [Symbol.toStringTag]() { return 'DeviceMemory'; }
}

interface PinnedMemoryConstructor extends MemoryConstructor {
    readonly prototype: Memory;
    new(byteLength?: number): Memory;
}

export class PinnedMemory extends (<PinnedMemoryConstructor> CUDA.PinnedMemory) {
    slice(start?: number, end?: number): PinnedMemory {
        [start, end] = clamp(this.byteLength, start, end);
        return super.slice(start, end - start) as PinnedMemory;
    }
    public get [Symbol.toStringTag]() { return 'PinnedMemory'; }
}

interface ManagedMemoryConstructor extends MemoryConstructor {
    readonly prototype: Memory;
    new(byteLength?: number): Memory;
}

export class ManagedMemory extends (<ManagedMemoryConstructor> CUDA.ManagedMemory) {
    slice(start?: number, end?: number): ManagedMemory {
        [start, end] = clamp(this.byteLength, start, end);
        return super.slice(start, end - start) as ManagedMemory;
    }
    public get [Symbol.toStringTag]() { return 'ManagedMemory'; }
}
