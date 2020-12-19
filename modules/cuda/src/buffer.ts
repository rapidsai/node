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
import { Memory, DeviceMemory, IpcHandle } from './memory';
import { MemoryData, TypedArray, BigIntArray, TypedArrayConstructor } from './interfaces';
import {
  clampSliceArgs as clamp,
  isNumber,
  isArrayBuffer,
  isArrayBufferView,
  isIterable,
  isMemoryLike,
  isArrayLike,
} from './util';

const {
  runtime: { cudaMemcpy },
} = CUDA;

/** @ignore */
type MemoryViewConstructor<T extends TypedArray | BigIntArray> = {
  readonly BYTES_PER_ELEMENT: number;
  new (length?: number): MemoryView<T>;
  new (values: Iterable<T[0]>): MemoryView<T>;
  new (buffer: ArrayLike<T[0]> | MemoryData, byteOffset?: number, length?: number): MemoryView<T>;
};

let allocateMemory = (byteLength: number): Memory => new DeviceMemory(byteLength);

export function setDefaultAllocator(allocate: (byteLength: number) => Memory) {
  allocateMemory = allocate;
}

/**
 * @summary A base class for typed arrays of values in CUDA device memory.
 */
abstract class MemoryView<T extends TypedArray | BigIntArray = any> implements ArrayBufferView {
  public static readonly BYTES_PER_ELEMENT: number;

  /**
   * @summary The size in bytes of each element in the array.
   */
  public readonly BYTES_PER_ELEMENT!: number;

  /**
   * @summary The {@link Memory `Memory`} instance referenced by the view.
   */
  public readonly buffer!: Memory;

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

  [index: number]: T[0];

  /**
   * @summary The constructor of this array's corresponding JS TypedArray.
   */
  public readonly TypedArray!: TypedArrayConstructor<T>;

  /**
   * @summary The length of the array.
   */
  public readonly [Symbol.species]!: MemoryViewConstructor<T>;

  constructor(length?: number);
  constructor(arrayOrArrayBuffer: Iterable<T[0]> | ArrayLike<T[0]> | MemoryData);
  constructor(buffer: ArrayLike<T[0]> | MemoryData, byteOffset: number, length?: number);
  constructor() {
    // eslint-disable-next-line prefer-const, prefer-rest-params
    let [buffer, byteOffset, length] = arguments;
    Object.assign(this, asMemory(buffer, this.TypedArray));
    switch (arguments.length) {
      // @ts-ignore
      case 3:
        this.length = Math.max(+length, 0) || 0;
        this.byteLength = this.length * this.BYTES_PER_ELEMENT;
      // @ts-ignore
      // eslint-disable-next-line no-fallthrough
      case 2:
        this.byteOffset = Math.max(+byteOffset, 0) || 0;
        break;
    }
  }

  public copyFrom(source: MemoryData, start?: number) {
    this.set(source, start);
    return this;
  }

  public copyInto(target: MemoryData, start?: number) {
    if (target instanceof MemoryView) {
      target.set(this, start);
    } else if (isArrayBuffer(target)) {
      cudaMemcpy(target, this, Math.min(this.byteLength, target.byteLength));
    } else if (isArrayBufferView(target)) {
      // eslint-disable-next-line prefer-const
      let [offset, size] = clamp(this.length, start);
      size = Math.min(size * this.BYTES_PER_ELEMENT, target.byteLength);
      cudaMemcpy(target, this.subarray(offset), size);
    }
    return this;
  }

  /**
   * Copies the underlying CUDA memory into a JavaScript typed array.
   * @returns A JavaScript typed array copy of the underlying CUDA memory.
   */
  public toArray(): T {
    const target = new this.TypedArray(this.length);
    this.copyInto(target);
    return target;
  }

  /**
   * Sets a value or an array of values.
   * @param array A typed or untyped array of values to set.
   * @param start The index in the current array at which the values are to be written.
   */
  public set(array: MemoryData | ArrayLike<T[0]>, start?: number) {
    // eslint-disable-next-line prefer-const
    let [offset, size] = clamp(this.length, start);
    const source = asMemoryView(array, this.TypedArray);
    size = Math.min(size * this.BYTES_PER_ELEMENT, source.byteLength);
    cudaMemcpy(this.subarray(offset), source, size);
  }

  /**
   * Returns the this object after filling the section identified by start and end with value.
   * @param value value to fill array section with.
   * @param start index to start filling the array at. If start is negative, it is treated as
   * length+start where length is the length of the array.
   * @param end index to stop filling the array at. If end is negative, it is treated as
   * length+end.
   */
  public fill(value: T[0], start?: number, end?: number) {
    [start, end] = clamp(this.length, start, end);
    this.set(new this.TypedArray(end - start).fill(<never>value), start);
    return this;
  }

  /**
   * Returns a section of an array.
   * @param start The beginning of the specified portion of the array.
   * @param end The end of the specified portion of the array. This is exclusive of the element at the index 'end'.
   */
  public slice(start?: number, end?: number) {
    [start, end] = clamp(this.length, start, end);
    return new this[Symbol.species](
      this.buffer.slice(
        this.byteOffset + start * this.BYTES_PER_ELEMENT,
        this.byteOffset + end * this.BYTES_PER_ELEMENT,
      ),
    );
  }

  /**
   * Creates a new MemoryView view over the underlying Memory of this array,
   * referencing the elements at begin, inclusive, up to end, exclusive.
   * @param begin The index of the beginning of the array.
   * @param end The index of the end of the array.
   */
  public subarray(begin?: number, end?: number) {
    [begin, end] = clamp(this.length, begin, end);
    return new this[Symbol.species](
      this.buffer,
      this.byteOffset + begin * this.BYTES_PER_ELEMENT,
      end - begin,
    );
  }

  public get [Symbol.toStringTag]() {
    return this.constructor.name;
  }
  public [Symbol.for('nodejs.util.inspect.custom')]() {
    return this.toString();
  }
  public toString() {
    return `${this[Symbol.toStringTag]} ${JSON.stringify({
      'length': this.length,
      'byteOffset': this.byteOffset,
      'byteLength': this.byteLength,
      'device': this.buffer.device,
      'type': this.buffer[Symbol.toStringTag],
    })}`;
  }

  /**
   * Gets an IpcHandle for the underlying CUDA device memory.
   */
  public getIpcHandle() {
    if (!(this.buffer instanceof DeviceMemory)) {
      throw new Error(`${this[Symbol.toStringTag]}'s buffer must be an instance of DeviceMemory`);
    }
    return new IpcHandle(this.buffer, this.byteOffset);
  }
}

Object.setPrototypeOf(
  MemoryView.prototype,
  new Proxy(
    {},
    {
      get(target: any, p: any, receiver: any) {
        let i: number = p;
        switch (typeof p) {
          // @ts-ignore
          case 'string':
            if (isNaN((i = +p))) {
              break;
            }
          // eslint-disable-next-line no-fallthrough
          case 'number':
            if (i > -1 && i < receiver.length) {
              const { byteOffset, BYTES_PER_ELEMENT, E } = receiver;
              // eslint-disable-next-line @typescript-eslint/restrict-plus-operands
              receiver.byteOffset = byteOffset + i * BYTES_PER_ELEMENT;
              cudaMemcpy(E, receiver, BYTES_PER_ELEMENT);
              receiver.byteOffset = byteOffset;
              return E[0];
            }
            return undefined;
        }
        return Reflect.get(target, p, receiver);
      },
      set(target: any, p: any, value: any, receiver: any) {
        let i: number = p;
        switch (typeof p) {
          // @ts-ignore
          case 'string':
            if (isNaN((i = +p))) {
              break;
            }
          // eslint-disable-next-line no-fallthrough
          case 'number':
            if (i > -1 && i < receiver.length) {
              const { byteOffset, BYTES_PER_ELEMENT, E } = receiver;
              // eslint-disable-next-line @typescript-eslint/restrict-plus-operands
              receiver.byteOffset = byteOffset + i * BYTES_PER_ELEMENT;
              E[0] = value;
              cudaMemcpy(receiver, E, BYTES_PER_ELEMENT);
              receiver.byteOffset = byteOffset;
              return true;
            }
        }
        return Reflect.set(target, p, value, receiver);
      },
    },
  ),
);

/** @ignore */ (<any>MemoryView.prototype)[Symbol.species] = MemoryView;
/** @ignore */ (<any>MemoryView.prototype).TypedArray = Uint8ClampedArray;
/** @ignore */ (<any>MemoryView.prototype).E = new Uint8ClampedArray(8);
/** @ignore */ (<any>MemoryView.prototype).BYTES_PER_ELEMENT = Uint8ClampedArray.BYTES_PER_ELEMENT;

/** @summary A typed array of twos-complement 8-bit signed integers in CUDA memory. */
export class Int8Buffer extends MemoryView<Int8Array> {
  public static readonly TypedArray: TypedArrayConstructor<Int8Array> = Int8Array;
}
/** @summary A typed array of twos-complement 16-bit signed integers in CUDA memory. */
export class Int16Buffer extends MemoryView<Int16Array> {
  public static readonly TypedArray: TypedArrayConstructor<Int16Array> = Int16Array;
}
/** @summary A typed array of twos-complement 32-bit signed integers in CUDA memory. */
export class Int32Buffer extends MemoryView<Int32Array> {
  public static readonly TypedArray: TypedArrayConstructor<Int32Array> = Int32Array;
}
/** @summary A typed array of 8-bit unsigned integers in CUDA memory. */
export class Uint8Buffer extends MemoryView<Uint8Array> {
  public static readonly TypedArray: TypedArrayConstructor<Uint8Array> = Uint8Array;
}
/** @summary A typed array of 8-bit unsigned integers clamped to 0-255 in CUDA memory. */
export class Uint8ClampedBuffer extends MemoryView<Uint8ClampedArray> {
  public static readonly TypedArray: TypedArrayConstructor<Uint8ClampedArray> = Uint8ClampedArray;
}
/** @summary A typed array of 16-bit unsigned integers in CUDA memory. */
export class Uint16Buffer extends MemoryView<Uint16Array> {
  public static readonly TypedArray: TypedArrayConstructor<Uint16Array> = Uint16Array;
}
/** @summary A typed array of 32-bit unsigned integers in CUDA memory. */
export class Uint32Buffer extends MemoryView<Uint32Array> {
  public static readonly TypedArray: TypedArrayConstructor<Uint32Array> = Uint32Array;
}
/** @summary A typed array of 32-bit floating point numbers in CUDA memory. */
export class Float32Buffer extends MemoryView<Float32Array> {
  public static readonly TypedArray: TypedArrayConstructor<Float32Array> = Float32Array;
}
/** @summary A typed array of 64-bit floating point numbers values in CUDA memory. */
export class Float64Buffer extends MemoryView<Float64Array> {
  public static readonly TypedArray: TypedArrayConstructor<Float64Array> = Float64Array;
}
/** @summary A typed array of 64-bit signed integers in CUDA memory. */
export class Int64Buffer extends MemoryView<BigInt64Array> {
  public static readonly TypedArray: TypedArrayConstructor<BigInt64Array> = BigInt64Array;
}
/** @summary A typed array of 64-bit unsigned integers in CUDA memory. */
export class Uint64Buffer extends MemoryView<BigUint64Array> {
  public static readonly TypedArray: TypedArrayConstructor<BigUint64Array> = BigUint64Array;
}

[
  { 0: Int8Buffer, 1: Int8Array },
  { 0: Int16Buffer, 1: Int16Array },
  { 0: Int32Buffer, 1: Int32Array },
  { 0: Uint8Buffer, 1: Uint8Array },
  { 0: Uint8ClampedBuffer, 1: Uint8ClampedArray },
  { 0: Uint16Buffer, 1: Uint16Array },
  { 0: Uint32Buffer, 1: Uint32Array },
  { 0: Float32Buffer, 1: Float32Array },
  { 0: Float64Buffer, 1: Float64Array },
  { 0: Int64Buffer, 1: BigInt64Array },
  { 0: Uint64Buffer, 1: BigUint64Array },
].forEach(({ 0: MemoryViewCtor, 1: TypedArrayCtor }) => {
  (<any>MemoryViewCtor.prototype).TypedArray = TypedArrayCtor;
  (<any>MemoryViewCtor.prototype)[Symbol.species] = MemoryViewCtor;
  (<any>MemoryViewCtor).BYTES_PER_ELEMENT = TypedArrayCtor.BYTES_PER_ELEMENT;
  (<any>MemoryViewCtor.prototype).BYTES_PER_ELEMENT = TypedArrayCtor.BYTES_PER_ELEMENT;
  (<any>MemoryViewCtor.prototype).E = new TypedArrayCtor((<any>MemoryView.prototype).E.buffer);
});

/** @internal */
function asMemory<T extends TypedArray | BigIntArray>(
  source: number | Iterable<T[0]> | ArrayLike<T[0]> | MemoryData,
  TypedArray: TypedArrayConstructor<T>,
) {
  let byteOffset = 0;
  let byteLength = 0;
  let buffer: Memory;
  if (isNumber(source)) {
    byteLength = source * TypedArray.BYTES_PER_ELEMENT;
    buffer = allocateMemory(source * TypedArray.BYTES_PER_ELEMENT);
  } else if (source instanceof MemoryView) {
    byteLength = source.byteLength;
    buffer = source.buffer.slice(source.byteOffset, byteLength);
  } else if (isMemoryLike(source)) {
    buffer = source;
    byteLength = source.byteLength;
  } else if (isArrayBuffer(source)) {
    byteLength = source.byteLength;
    buffer = allocateMemory(byteLength);
    cudaMemcpy(buffer, source, byteLength);
  } else if (isArrayBufferView(source)) {
    byteLength = source.byteLength;
    buffer = allocateMemory(byteLength);
    cudaMemcpy(buffer, source, byteLength);
  } else if (isIterable(source)) {
    const b = new TypedArray(source).buffer;
    byteLength = b.byteLength;
    buffer = allocateMemory(byteLength);
    cudaMemcpy(buffer, b, byteLength);
  } else if (isArrayLike(source)) {
    const b = TypedArray.from(source).buffer;
    byteLength = b.byteLength;
    buffer = allocateMemory(byteLength);
    cudaMemcpy(buffer, b, byteLength);
  } else if ('buffer' in source && 'byteOffset' in source && 'byteLength' in source) {
    buffer = source['buffer'];
    byteLength = source['byteLength'];
    byteOffset = source['byteOffset'];
  } else {
    byteOffset = 0;
    byteLength = 0;
    buffer = allocateMemory(0);
  }
  return { buffer, byteLength, byteOffset, length: byteLength / TypedArray.BYTES_PER_ELEMENT };
}

/** @internal */
function asMemoryView<T extends TypedArray | BigIntArray>(
  source: Iterable<T[0]> | ArrayLike<T[0]> | MemoryData,
  TypedArray: TypedArrayConstructor<T>,
) {
  if (source instanceof MemoryView) {
    return source;
  }
  switch (TypedArray.name) {
    case 'Int8Array':
      return new Int8Buffer(source as MemoryData);
    case 'Int16Array':
      return new Int16Buffer(source as MemoryData);
    case 'Int32Array':
      return new Int32Buffer(source as MemoryData);
    case 'Uint8Array':
      return new Uint8Buffer(source as MemoryData);
    case 'Uint8ClampedArray':
      return new Uint8ClampedBuffer(source as MemoryData);
    case 'Uint16Array':
      return new Uint16Buffer(source as MemoryData);
    case 'Uint32Array':
      return new Uint32Buffer(source as MemoryData);
    case 'Float32Array':
      return new Float32Buffer(source as MemoryData);
    case 'Float64Array':
      return new Float64Buffer(source as MemoryData);
    case 'BigInt64Array':
      return new Int64Buffer(source as MemoryData);
    case 'BigUint64Array':
      return new Uint64Buffer(source as MemoryData);
  }
  throw new Error('Unknown dtype');
}
