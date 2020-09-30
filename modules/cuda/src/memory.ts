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
import { CUDABuffer } from './addon'; 
import { CUDAMemoryType } from './cuda';
import { CUDAPointerAttribute } from './cuda';
import { cachedEnumLookup } from './util';

export class CUDAMemory implements ArrayBufferView {

    public static alloc(size: number) { return new CUDAMemory(CUDA.mem.alloc(size)); }
    public static allocHost(size: number) { return new CUDAMemory(CUDA.mem.allocHost(size)); }

    public static as(source: CUDAMemory | ArrayBufferLike | ArrayBufferView | ArrayLike<number>) {
        if (!(source instanceof CUDAMemory)) {
            let buffer: ArrayBuffer;
            let byteOffset = 0;
            let byteLength = 0;
            if ((source instanceof ArrayBuffer) || (source instanceof SharedArrayBuffer)) {
                byteLength = (buffer = source).byteLength;
            } else if (!ArrayBuffer.isView(source)) {
                ({buffer, byteLength} = Uint8Array.from(source));
            } else {
                buffer = source.buffer;
                byteOffset = source.byteOffset;
                byteLength = source.byteLength;
            }
            source = new CUDAMemory(buffer, byteOffset, byteLength);
        }
        return source as CUDAMemory;
    }

    protected constructor(buffer: CUDABuffer | ArrayBufferLike, byteOffset = 0, byteLength = buffer.byteLength - byteOffset) {
        this.buffer = buffer;
        this.byteOffset = byteOffset;
        this.byteLength = byteLength;
    }

    public readonly buffer: any;
    public readonly byteOffset: number;
    public readonly byteLength: number;

    protected get id() { return this.buffer; };
    
    public get length() { return this.byteLength; }
    public isHost() { return this.getMemoryType() === CUDAMemoryType.HOST; }
    public isDevice() { return this.getMemoryType() === CUDAMemoryType.DEVICE; }
    public isManaged() { return this.getMemoryType() === CUDAMemoryType.MANAGED; }
    public isUnregistered() { return this.getMemoryType() === CUDAMemoryType.UNREGISTERED; }

    public register(flags: CUDA.CUDAMemHostRegisterFlag = CUDA.CUDAMemHostRegisterFlag.DEFAULT) {
        if (this.isUnregistered()) {
            delete (<any> this)['_memoryType'];
            CUDA.mem.hostRegister(this.buffer, flags);
        }
        return this;
    }

    public unregister() {
        if (this.isHost()) {
            delete (<any> this)['_memoryType'];
            CUDA.mem.hostUnregister(this.buffer);
        }
        return this;
    }

    public copyFrom(source: CUDAMemory | ArrayLike<number>, start?: number) {
        this.set(source, start);
        return this;
    }

    public async copyFromAsync(source: CUDAMemory | ArrayLike<number>, start?: number, stream: CUDA.CUstream = 0) {
        await this.setAsync(source, start, stream);
        return this;
    }

    public copyInto(target: CUDAMemory | ArrayBufferLike | ArrayBufferView, start?: number) {
        CUDAMemory.as(target).set(this, start);
        return this;
    }

    public async copyIntoAsync(target: CUDAMemory | ArrayBufferLike | ArrayBufferView, start?: number, stream: CUDA.CUstream = 0) {
        await CUDAMemory.as(target).setAsync(this, start, stream);
        return this;
    }

    public set(values: CUDAMemory | ArrayLike<number>, start?: number): void {
        const source = CUDAMemory.as(values);
        const sourceLength = source.byteLength;
        const [offset, length] = clamp(this, start);
        CUDA.mem.cpy(this.buffer, offset + this.byteOffset,
                     source.buffer, source.byteOffset,
                     Math.min(length, sourceLength));
    }

    public async setAsync(values: CUDAMemory | ArrayLike<number>, start?: number, stream: CUDA.CUstream = 0): Promise<void> {
        const source = CUDAMemory.as(values);
        const sourceLength = source.byteLength;
        const [offset, length] = clamp(this, start);
        await CUDA.mem.cpyAsync(this.buffer, offset + this.byteOffset,
                                source.buffer, source.byteOffset,
                                Math.min(length, sourceLength),
                                stream);
    }

    /** @inheritdoc */
    public fill(value: number, start?: number, end?: number) {
        [start, end] = clamp(this, start, end);
        CUDA.mem.set(this.buffer, this.byteOffset + start, value, (end - start));
        return this;
    }

    public async fillAsync(value: number, start?: number, end?: number, stream: CUDA.CUstream = 0): Promise<this> {
        [start, end] = clamp(this, start, end);
        await CUDA.mem.setAsync(this.buffer, this.byteOffset + start, value, (end - start), stream);
        return this;
    }
}

export interface CUDAMemory {
    /**
     * A boolean indicating whether the pointer points to managed memory or not.
     */
    getIsManaged(): boolean;
    /**
     * Maximum number of threads per block.
     */
    getMemoryType(): CUDAMemoryType;
}

CUDAMemory.prototype.getIsManaged = cachedEnumLookup('isManaged', CUDAPointerAttribute.IS_MANAGED, CUDA.mem.getPointerAttribute);
CUDAMemory.prototype.getMemoryType = cachedEnumLookup('memoryType', CUDAPointerAttribute.MEMORY_TYPE, CUDA.mem.getPointerAttribute);

function clamp(mem: CUDAMemory, start?: number, end?: number): [number, number] {
    // Adjust args similar to Array.prototype.slice. Normalize begin/end to
    // clamp between 0 and length, and wrap around on negative indices, e.g.
    // slice(-1, 5) or slice(5, -1)
    let len = mem.byteLength - mem.byteOffset;
    let lhs = typeof start === 'number' ? start : 0;
    let rhs = typeof end === 'number' ? end : len;
    // wrap around on negative start/end positions
    if (lhs < 0) { lhs = ((lhs % len) + len) % len; }
    if (rhs < 0) { rhs = ((rhs % len) + len) % len; }
    /* enforce l <= r and r <= count */
    return rhs < lhs ? [rhs, lhs] : [lhs, rhs > len ? len : rhs];
}
