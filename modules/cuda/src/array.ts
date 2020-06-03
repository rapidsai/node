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

import { CUDABuffer, mem } from './cuda';
import { isNumber, isArrayBuffer, isArrayBufferView, isCUDABuffer, isIterable } from './util';

interface CUDAArray<T extends number | bigint = number> {
    /**
     * @summary The constructor of this array's corresponding JS TypedArray.
     */
    readonly T: TypedArrayConstructor<T>;
}

/**
 * @summary A base class for typed arrays of values in CUDA device memory.
 */
abstract class CUDAArray<T extends number | bigint = number> {

    /**
     * @summary The size in bytes of each element in the array.
     */
    public readonly BYTES_PER_ELEMENT: number;

    /**
     * @summary The {@link CUDABuffer `CUDABuffer`}  instance referenced by the array.
     */
    // @ts-ignore
    public readonly buffer: CUDABuffer;

    /**
     * @summary The offset in bytes of the array.
     */
    // @ts-ignore
    public readonly byteOffset: number;

    /**
     * @summary The length in bytes of the array.
     */
    // @ts-ignore
    public readonly byteLength: number;

    /**
     * @summary The length of the array.
     */
    // @ts-ignore
    public readonly length: number;

    [index: number]: number | undefined;

    constructor(length?: number);
    constructor(arrayOrArrayBuffer: Iterable<T> | ArrayBufferLike | CUDABuffer);
    constructor(buffer: ArrayBufferLike | CUDABuffer, byteOffset: number, length?: number);
    constructor() {
        let [buffer, byteOffset, length] = arguments;
        Object.assign(this, asCUDABuffer(buffer, this.T));
        this.BYTES_PER_ELEMENT = this.T.BYTES_PER_ELEMENT;
        switch (arguments.length) {
            // @ts-ignore
            case 3:
                this.length = length = Math.max(+length, 0) || 0;
                this.byteLength = length * this.BYTES_PER_ELEMENT;
            // @ts-ignore
            case 2: this.byteOffset = Math.max(+byteOffset, 0) || 0;
            default: break;
        }
        return new Proxy(this, CUDABufferViewProxyHandler);
    }
}

/**
 * @summary A typed array of twos-complement 8-bit signed integers in CUDA device memory.
 */
export class CUDAInt8Array extends CUDAArray<number> {}
/**
 * @summary A typed array of twos-complement 16-bit signed integers in CUDA device memory.
 */
export class CUDAInt16Array extends CUDAArray<number> {}
/**
 * @summary A typed array of twos-complement 32-bit signed integers in CUDA device memory.
 */
export class CUDAInt32Array extends CUDAArray<number> {}
/**
 * @summary A typed array of 8-bit unsigned integers in CUDA device memory.
 */
export class CUDAUint8Array extends CUDAArray<number> {}
/**
 * @summary A typed array of 8-bit unsigned integers clamped to 0-255 in CUDA device memory.
 */
export class CUDAUint8ClampedArray extends CUDAArray<number> {}
/**
 * @summary A typed array of 16-bit unsigned integers in CUDA device memory.
 */
export class CUDAUint16Array extends CUDAArray<number> {}
/**
 * @summary A typed array of 32-bit unsigned integers in CUDA device memory.
 */
export class CUDAUint32Array extends CUDAArray<number> {}
/**
 * @summary A typed array of 32-bit floating point numbers in CUDA device memory.
 */
export class CUDAFloat32Array extends CUDAArray<number> {}
/**
 * @summary A typed array of 64-bit floating point numbers values in CUDA device memory.
 */
export class CUDAFloat64Array extends CUDAArray<number> {}
/**
 * @summary A typed array of 64-bit signed integers in CUDA device memory.
 */
export class CUDAInt64Array extends CUDAArray<bigint> {}
/**
 * @summary A typed array of 64-bit unsigned integers in CUDA device memory.
 */
export class CUDAUint64Array extends CUDAArray<bigint> {}


/** @ignore */ (<any> CUDAArray.prototype).T = Uint8ClampedArray;
/** @ignore */ (<any> CUDAInt8Array.prototype).T = Int8Array;
/** @ignore */ (<any> CUDAInt16Array.prototype).T = Int16Array;
/** @ignore */ (<any> CUDAInt32Array.prototype).T = Int32Array;
/** @ignore */ (<any> CUDAUint8Array.prototype).T = Uint8Array;
/** @ignore */ (<any> CUDAUint8ClampedArray.prototype).T = Uint8ClampedArray;
/** @ignore */ (<any> CUDAUint16Array.prototype).T = Uint16Array;
/** @ignore */ (<any> CUDAUint32Array.prototype).T = Uint32Array;
/** @ignore */ (<any> CUDAFloat32Array.prototype).T = Float32Array;
/** @ignore */ (<any> CUDAFloat64Array.prototype).T = Float64Array;
/** @ignore */ (<any> CUDAInt64Array.prototype).T = BigInt64Array;
/** @ignore */ (<any> CUDAUint64Array.prototype).T = BigUint64Array;

/** @ignore */ (<any> CUDAArray.prototype).E = new CUDAArray.prototype.T(8);
/** @ignore */ (<any> CUDAInt8Array.prototype).E = new CUDAInt8Array.prototype.T((<any> CUDAArray.prototype).E.buffer);
/** @ignore */ (<any> CUDAInt16Array.prototype).E = new CUDAInt16Array.prototype.T((<any> CUDAArray.prototype).E.buffer);
/** @ignore */ (<any> CUDAInt32Array.prototype).E = new CUDAInt32Array.prototype.T((<any> CUDAArray.prototype).E.buffer);
/** @ignore */ (<any> CUDAUint8Array.prototype).E = new CUDAUint8Array.prototype.T((<any> CUDAArray.prototype).E.buffer);
/** @ignore */ (<any> CUDAUint8ClampedArray.prototype).E = new CUDAUint8ClampedArray.prototype.T((<any> CUDAArray.prototype).E.buffer);
/** @ignore */ (<any> CUDAUint16Array.prototype).E = new CUDAUint16Array.prototype.T((<any> CUDAArray.prototype).E.buffer);
/** @ignore */ (<any> CUDAUint32Array.prototype).E = new CUDAUint32Array.prototype.T((<any> CUDAArray.prototype).E.buffer);
/** @ignore */ (<any> CUDAFloat32Array.prototype).E = new CUDAFloat32Array.prototype.T((<any> CUDAArray.prototype).E.buffer);
/** @ignore */ (<any> CUDAFloat64Array.prototype).E = new CUDAFloat64Array.prototype.T((<any> CUDAArray.prototype).E.buffer);
/** @ignore */ (<any> CUDAInt64Array.prototype).E = new CUDAInt64Array.prototype.T((<any> CUDAArray.prototype).E.buffer);
/** @ignore */ (<any> CUDAUint64Array.prototype).E = new CUDAUint64Array.prototype.T((<any> CUDAArray.prototype).E.buffer);

/** @internal */ 
const CUDABufferViewProxyHandler: ProxyHandler<CUDAArray<any>> = {
    has(target: CUDAArray, p: any): boolean {
        if (isNumber(p)) {
            return p > -1 && p < target.length;
        }
        return Reflect.has(target, p);
    },
    get(target: CUDAArray, p: any, receiver: any): any {
        switch (typeof p) {
            // @ts-ignore
            case 'string':
                if (isNaN(+p)) { break; }
            case 'number':
                if ((p = +p) > -1 && p < receiver.length) {
                    p = p * receiver.BYTES_PER_ELEMENT + receiver.byteOffset;
                    mem.cpy(receiver.E.buffer, 0, receiver.buffer, p, receiver.BYTES_PER_ELEMENT);
                    return receiver.E[0];
                }
                return undefined;
        }
        return Reflect.get(target, p, receiver);
    },
    set(target: CUDAArray, p: any, value: any, receiver: any): boolean {
        switch (typeof p) {
            // @ts-ignore
            case 'string':
                if (isNaN(+p)) { break; }
            case 'number':
                const vt = typeof value;
                if ((vt !== 'number' && vt !== 'bigint') || (p = +p) < 0 || p > receiver.length - 1) {
                    return false;
                }
                mem.set(receiver.buffer, p * receiver.BYTES_PER_ELEMENT + receiver.byteOffset, value, 1);
                return true;
        }
        return Reflect.set(target, p, value, receiver);
    }
};

/** @ignore */
type FloatArray = Float32Array | Float64Array;
/** @ignore */
type IntArray = Int8Array | Int16Array | Int32Array;
/** @ignore */
type UintArray = Uint8Array | Uint16Array | Uint32Array | Uint8ClampedArray;
/** @ignore */
type BigIntArray = BigInt64Array | BigUint64Array;
/** @ignore */
type TypedArray = FloatArray | IntArray | UintArray;
/** @ignore */
type TypedArrayConstructor<T extends number | bigint> = {
    readonly BYTES_PER_ELEMENT: number;
    new(length?: number): T extends number ? TypedArray : BigIntArray;
    new(values: Iterable<T>): T extends number ? TypedArray : BigIntArray;
    new(buffer: ArrayBufferLike, byteOffset?: number, length?: number): T extends number ? TypedArray : BigIntArray;
};

/** @internal */ 
function asCUDABuffer<T extends number | bigint>(x: number | Iterable<T> | ArrayBufferLike | CUDABuffer | CUDAArray, T: TypedArrayConstructor<T>) {
    let byteLength = 0;
    let buffer = mem.alloc(0);
    if (isNumber(x)) {
        byteLength = x * T.BYTES_PER_ELEMENT;
        buffer = mem.alloc(x * T.BYTES_PER_ELEMENT);
    } else if (isCUDABuffer(x)) {
        buffer = x;
        byteLength = x.byteLength;
    } else if (x instanceof CUDAArray) {
        byteLength = x.byteLength;
        buffer = x.buffer.slice(x.byteOffset, byteLength);
    } else if (isArrayBuffer(x)) {
        byteLength = x.byteLength;
        buffer = mem.alloc(byteLength);
        mem.cpy(buffer, 0, x, 0, byteLength);
    } else if (isArrayBufferView(x)) {
        byteLength = x.byteLength;
        buffer = mem.alloc(byteLength);
        mem.cpy(buffer, 0, x.buffer, 0, byteLength);
    } else if (isIterable(x)) {
        const b = new T(x).buffer;
        byteLength = b.byteLength;
        buffer = mem.alloc(byteLength);
        mem.cpy(buffer, 0, b, 0, byteLength);
    }
    return { buffer, byteLength, byteOffset: 0, length: byteLength / T.BYTES_PER_ELEMENT };
};
