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

export class DeviceMemory extends CUDA.DeviceMemory {
    public get [Symbol.toStringTag]() { return 'DeviceMemory'; }
}

export class PinnedMemory extends CUDA.PinnedMemory {
    public get [Symbol.toStringTag]() { return 'PinnedMemory'; }
}

export class ManagedMemory extends CUDA.ManagedMemory {
    public get [Symbol.toStringTag]() { return 'ManagedMemory'; }
}

export class IpcMemory extends CUDA.IpcMemory {
    constructor(ipch: IpcHandle | Uint8Array | Buffer | Array<number>) {
        if (Array.isArray(ipch)) {
            ipch = Buffer.from(ipch);
        } else if (ipch instanceof IpcHandle) {
            ipch = Buffer.from(ipch.handle);
        }
        super(ipch);
    }
    public get [Symbol.toStringTag]() { return 'IpcMemory'; }
}

export class IpcHandle extends CUDA.IpcHandle {
    constructor(deviceMemory: DeviceMemory, byteOffset: number = 0) {
        super(deviceMemory);
        this.byteOffset = byteOffset;
    }
    public readonly byteOffset: number;
    public get [Symbol.toStringTag]() { return 'IpcHandle'; }
    public [Symbol.for('nodejs.util.inspect.custom')]() {
        return`${this[Symbol.toStringTag]} ${this.toString()}`;
    }
    public toString() { return JSON.stringify(this.toJSON()); }
    public toJSON() {
        return {
            device: this.device,
            byteOffset: this.byteOffset,
            handle: [...this.handle],
        };
    }
}
