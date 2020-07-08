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
import { isNumber, isArrayLike, isArrayBuffer, isArrayBufferView, isCUDABuffer, isIterable } from './util';

/**
 * @summary A base class for typed arrays of values in CUDA device memory.
 */
abstract class CUDAArray<T extends TypedArray | BigIntArray = any> {

    /**
     * @summary The size in bytes of each element in the array.
     */
    public readonly BYTES_PER_ELEMENT!: number;

    /**
     * @summary The {@link CUDABuffer `CUDABuffer`}  instance referenced by the array.
     */
    public readonly buffer!: CUDABuffer;

    /**
     * @summary The offset in bytes of the array.
     */
    public readonly byteOffset!: number;

    /**
     * @summary The length in bytes of the array.
     */
    public readonly byteLength!: number;

    /**
     * @summary The length of the array.
     */
    public readonly length!: number;

    [index: number]: T[0] | undefined;

    /**
     * @summary The constructor of this array's corresponding JS TypedArray.
     */
    protected readonly _T!: TypedArrayConstructor<T>;

    /**
     * @summary The length of the array.
     */
    public readonly [Symbol.species]!: CUDAArrayConstructor<T>;

    constructor(length?: number);
    constructor(arrayOrArrayBuffer: Iterable<T[0]> | ArrayLike<T[0]> | ArrayBufferLike | CUDABuffer);
    constructor(buffer: ArrayLike<T[0]> | ArrayBufferLike | CUDABuffer, byteOffset: number, length?: number);
    constructor() {
        let [buffer, byteOffset, length] = arguments;
        Object.assign(this, asCUDABuffer(buffer, this._T));
        switch (arguments.length) {
            // @ts-ignore
            case 3:
                this.length = length = Math.max(+length, 0) || 0;
                this.byteLength = length * this.BYTES_PER_ELEMENT;
            // @ts-ignore
            case 2: this.byteOffset = Math.max(+byteOffset, 0) || 0;
            default: break;
        }
    }

    public copyFrom(source: CUDAArray | ArrayLike<T[0]>, start?: number) {
        this.set(source, start);
        return this;
    }

    public copyInto(target: CUDAArray | ArrayBufferLike | ArrayBufferView, start?: number) {
        if (target instanceof CUDAArray) {
            target.set(this, start);
        } else if (isArrayBuffer(target)) {
            mem.cpy(
                target, 0,
                this.buffer, this.byteOffset,
                Math.min(this.byteLength, target.byteLength)
            );
        } else if (isArrayBufferView(target)) {
            const [offset, length] = clamp(target, start);
            const BPE = (<any> target).BYTES_PER_ELEMENT || 1;
            mem.cpy(
                target.buffer,
                target.byteOffset + (offset * BPE),
                this.buffer, this.byteOffset,
                Math.min(this.byteLength, length * BPE)
            );
        }
        return this;
    }

    /**
     * Sets a value or an array of values.
     * @param array A typed or untyped array of values to set.
     * @param offset The index in the current array at which the values are to be written.
     */
    public set(array: CUDAArray | ArrayLike<T[0]>, offset?: number) {
        const source = asCUDAArray(array, this._T);
        const [start, length] = clamp(this, offset);
        mem.cpy(
            this.buffer,
            this.byteOffset + (start * this.BYTES_PER_ELEMENT),
            source.buffer,
            source.byteOffset,
            Math.min(source.byteLength, length * this.BYTES_PER_ELEMENT)
        );
    }

    /**
     * Returns the this object after filling the section identified by start and end with value.
     * @param value value to fill array section with.
     * @param start index to start filling the array at. If start is negative, it is treated as
     * length+start where length is the length of the array.
     * @param end index to stop filling the array at. If end is negative, it is treated as
     * length+end.
     */
    public fill(value: number, start?: number, end?: number) {
        [start, end] = clamp(this, start, end);
        mem.set(this.buffer, this.byteOffset + start, value, end - start);
        return this;
    }

    /**
     * Returns a section of an array.
     * @param start The beginning of the specified portion of the array.
     * @param end The end of the specified portion of the array. This is exclusive of the element at the index 'end'.
     */
    public slice(start?: number, end?: number) {
        [start, end] = clamp(this, start, end);
        return new this[Symbol.species](this.buffer.slice(
            this.byteOffset + (start * this.BYTES_PER_ELEMENT),
            this.byteOffset + (end * this.BYTES_PER_ELEMENT)
        ));
    }

    /**
     * Gets a new CUDAArray view of the CUDABuffer store for this array,
     * referencing the elements at begin, inclusive, up to end, exclusive.
     * @param begin The index of the beginning of the array.
     * @param end The index of the end of the array.
     */
    public subarray(begin?: number, end?: number) {
        [begin, end] = clamp(this, begin, end);
        return new this[Symbol.species](
            this.buffer,
            this.byteOffset + (begin * this.BYTES_PER_ELEMENT),
            end - begin
        );
    }
}

Object.setPrototypeOf(CUDAArray.prototype, new Proxy({}, {
    get(target: {}, p: any, receiver: any) {
        switch (typeof p) {
            // @ts-ignore
            case 'string':
                if (isNaN(+p)) { break; }
            case 'number':
                if ((p = +p) > -1 && p < receiver.length) {
                    p = p * receiver.BYTES_PER_ELEMENT + receiver.byteOffset;
                    mem.cpy(receiver.ElementArray.buffer, 0, receiver.buffer, p, receiver.BYTES_PER_ELEMENT);
                    return receiver.ElementArray[0];
                }
                return undefined;
        }
        return Reflect.get(target, p, receiver);
    },
    set(target: {}, p: any, value: any, receiver: any) {
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
}));

/**
 * @summary A typed array of twos-complement 8-bit signed integers in CUDA device memory.
 */
export class CUDAInt8Array extends CUDAArray<Int8Array> {}
/**
 * @summary A typed array of twos-complement 16-bit signed integers in CUDA device memory.
 */
export class CUDAInt16Array extends CUDAArray<Int16Array> {}
/**
 * @summary A typed array of twos-complement 32-bit signed integers in CUDA device memory.
 */
export class CUDAInt32Array extends CUDAArray<Int32Array> {}
/**
 * @summary A typed array of 8-bit unsigned integers in CUDA device memory.
 */
export class CUDAUint8Array extends CUDAArray<Uint8Array> {}
/**
 * @summary A typed array of 8-bit unsigned integers clamped to 0-255 in CUDA device memory.
 */
export class CUDAUint8ClampedArray extends CUDAArray<Uint8ClampedArray> {}
/**
 * @summary A typed array of 16-bit unsigned integers in CUDA device memory.
 */
export class CUDAUint16Array extends CUDAArray<Uint16Array> {}
/**
 * @summary A typed array of 32-bit unsigned integers in CUDA device memory.
 */
export class CUDAUint32Array extends CUDAArray<Uint32Array> {}
/**
 * @summary A typed array of 32-bit floating point numbers in CUDA device memory.
 */
export class CUDAFloat32Array extends CUDAArray<Float32Array> {}
/**
 * @summary A typed array of 64-bit floating point numbers values in CUDA device memory.
 */
export class CUDAFloat64Array extends CUDAArray<Float64Array> {}
/**
 * @summary A typed array of 64-bit signed integers in CUDA device memory.
 */
export class CUDAInt64Array extends CUDAArray<BigInt64Array> {}
/**
 * @summary A typed array of 64-bit unsigned integers in CUDA device memory.
 */
export class CUDAUint64Array extends CUDAArray<BigUint64Array> {}

/** @internal */ 
function clamp(mem: any, start?: number, end?: number): [number, number] {
    // Adjust args similar to Array.prototype.slice. Normalize begin/end to
    // clamp between 0 and length, and wrap around on negative indices, e.g.
    // slice(-1, 5) or slice(5, -1)
    let len = mem.length || mem.byteLength;
    let lhs = typeof start === 'number' ? start : 0;
    let rhs = typeof end === 'number' ? end : len;
    // wrap around on negative start/end positions
    if (lhs < 0) { lhs = ((lhs % len) + len) % len; }
    if (rhs < 0) { rhs = ((rhs % len) + len) % len; }
    /* enforce l <= r and r <= count */
    return rhs < lhs ? [rhs, lhs] : [lhs, rhs > len ? len : rhs];
}

function asCUDAArray<T extends TypedArray | BigIntArray>(x: ArrayLike<T[0]> | CUDAArray<T>, T: TypedArrayConstructor<T>) {
    if (x instanceof CUDAArray) {
        return x;
    }
    switch (T.name) {
        case 'Int8Array': return new CUDAInt8Array(x as any);
        case 'Int16Array': return new CUDAInt16Array(x as any);
        case 'Int32Array': return new CUDAInt32Array(x as any);
        case 'Uint8Array': return new CUDAUint8Array(x as any);
        case 'Uint8ClampedArray': return new CUDAUint8ClampedArray(x as any);
        case 'Uint16Array': return new CUDAUint16Array(x as any);
        case 'Uint32Array': return new CUDAUint32Array(x as any);
        case 'Float32Array': return new CUDAFloat32Array(x as any);
        case 'Float64Array': return new CUDAFloat64Array(x as any);
        case 'BigInt64Array': return new CUDAInt64Array(x as any);
        case 'BigUint64Array': return new CUDAUint64Array(x as any);
    }
    throw new Error('Unknown dtype');
}

/** @internal */ 
function asCUDABuffer<T extends TypedArray | BigIntArray>(x: number | Iterable<T[0]> | ArrayBufferLike | CUDABuffer | CUDAArray, T: TypedArrayConstructor<T>) {
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
    } else if (isArrayLike(x)) {
        const b = T.from(x).buffer;
        byteLength = b.byteLength;
        buffer = mem.alloc(byteLength);
        mem.cpy(buffer, 0, b, 0, byteLength);
    }
    return { buffer, byteLength, byteOffset: 0, length: byteLength / T.BYTES_PER_ELEMENT };
}

/** @ignore */ (<any> CUDAArray.prototype)[Symbol.species] = CUDAArray;
/** @ignore */ (<any> CUDAArray.prototype).TypedArray = Uint8ClampedArray;
/** @ignore */ (<any> CUDAArray.prototype).ElementArray = new Uint8ClampedArray(8);
/** @ignore */ (<any> CUDAArray.prototype).BYTES_PER_ELEMENT = Uint8ClampedArray.BYTES_PER_ELEMENT;

[
    { 0: CUDAInt8Array, 1: Int8Array },
    { 0: CUDAInt16Array, 1: Int16Array },
    { 0: CUDAInt32Array, 1: Int32Array },
    { 0: CUDAUint8Array, 1: Uint8Array },
    { 0: CUDAUint8ClampedArray, 1: Uint8ClampedArray },
    { 0: CUDAUint16Array, 1: Uint16Array },
    { 0: CUDAUint32Array, 1: Uint32Array },
    { 0: CUDAFloat32Array, 1: Float32Array },
    { 0: CUDAFloat64Array, 1: Float64Array },
    { 0: CUDAInt64Array, 1: BigInt64Array },
    { 0: CUDAUint64Array, 1: BigUint64Array },
].forEach(({ 0: CUDAArrayCtor, 1: TypedArrayCtor }) => {
    (<any> CUDAArrayCtor.prototype).TypedArray = TypedArrayCtor;
    (<any> CUDAArrayCtor.prototype)[Symbol.species] = CUDAArrayCtor;
    (<any> CUDAArrayCtor.prototype).BYTES_PER_ELEMENT = TypedArrayCtor.BYTES_PER_ELEMENT;
    (<any> CUDAArrayCtor.prototype).ElementArray = new TypedArrayCtor((<any> CUDAArray.prototype).ElementArray.buffer);
});

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
type TypedArrayConstructor<T extends TypedArray | BigIntArray> = {
    readonly BYTES_PER_ELEMENT: number;
    new(length?: number): T;
    new(values: Iterable<T[0]> | ArrayLike<T[0]>): T;
    new(buffer: ArrayBufferLike, byteOffset?: number, length?: number): T;
    from(arrayLike: ArrayLike<T[0]>): T;
};

/** @ignore */
type CUDAArrayType<T extends TypedArray | BigIntArray> =
    T extends Int8Array ? CUDAInt8Array : 
    T extends Int16Array ? CUDAInt16Array : 
    T extends Int32Array ? CUDAInt32Array : 
    T extends Uint8Array ? CUDAUint8Array : 
    T extends Uint8ClampedArray ? CUDAUint8ClampedArray : 
    T extends Uint16Array ? CUDAUint16Array : 
    T extends Uint32Array ? CUDAUint32Array : 
    T extends Float32Array ? CUDAFloat32Array : 
    T extends Float64Array ? CUDAFloat64Array : 
    T extends BigInt64Array ? CUDAInt64Array : 
    T extends BigUint64Array ? CUDAUint64Array : never;

/** @ignore */
type CUDAArrayConstructor<T extends TypedArray | BigIntArray> = {
    readonly BYTES_PER_ELEMENT: number;
    new(length?: number): CUDAArrayType<T>;
    new(values: Iterable<T[0]>): CUDAArrayType<T>;
    new(buffer: ArrayLike<T[0]> | ArrayBufferLike | CUDABuffer, byteOffset?: number, length?: number): CUDAArrayType<T>;
};
