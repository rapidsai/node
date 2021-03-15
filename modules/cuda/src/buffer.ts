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
import {BigIntArray, MemoryData, TypedArray, TypedArrayConstructor} from './interfaces';
import {DeviceMemory, IpcHandle, Memory} from './memory';
import {
  clampRange,
  isArrayBufferLike,
  isArrayBufferView,
  isArrayLike,
  isIterable,
  isMemoryLike,
  isNumber,
  isObject
} from './util';

const {min, max}                          = Math;
const {runtime: {cudaMemcpy, cudaMemset}} = CUDA;

/** @ignore */
// clang-format off
type MemoryViewOf<T extends TypedArray|BigIntArray> =
    T extends Int8Array         ? Int8Buffer
  : T extends Int16Array        ? Int16Buffer
  : T extends Int32Array        ? Int32Buffer
  : T extends BigInt64Array     ? Int64Buffer
  : T extends Uint8Array        ? Uint8Buffer
  : T extends Uint8ClampedArray ? Uint8ClampedBuffer
  : T extends Uint16Array       ? Uint16Buffer
  : T extends Uint32Array       ? Uint32Buffer
  : T extends BigUint64Array    ? Uint64Buffer
  : T extends Float32Array      ? Float32Buffer
  : T extends Float64Array      ? Float64Buffer
  : never;
// clang-format on

export type CUDAMemoryView = Int8Buffer|Int16Buffer|Int32Buffer|Int64Buffer|Uint8Buffer|
  Uint8ClampedBuffer|Uint16Buffer|Uint32Buffer|Uint64Buffer|Float32Buffer|Float64Buffer;

/** @ignore */
export type MemoryViewConstructor<T extends TypedArray|BigIntArray> = {
  readonly BYTES_PER_ELEMENT: number,
  readonly TypedArray: TypedArrayConstructor<T>,
  new (length?: number): MemoryViewOf<T>,
  new (values: Iterable<T[0]>): MemoryViewOf<T>,
  new (buffer: ArrayLike<T[0]>|MemoryData, byteOffset?: number, length?: number): MemoryViewOf<T>,
};

const allocateDeviceMemory = (byteLength: number): Memory => new DeviceMemory(byteLength);

let allocateMemory = allocateDeviceMemory;

/**
 * @summary A function to override the default device memory allocation behavior.
 * The supplied function will be called to create the underlying {@link Memory `Memory`} instances
 * when constructing one of the {@link CUDAMemoryView `CUDAMemoryView`} in JavaScript.
 *
 * @example
 * ```typescript
 * import {
 *   DeviceMemory,
 *   ManagedMemory,
 *   Float32Buffer,
 *   setDefaultAllocator
 * } from '@nvidia/cuda';
 *
 * // The default allocator creates `DeviceMemory` instances,
 * // which can only be accessed directly from the GPU device.
 * // An expensive copy from GPU to CPU memory must be performed
 * // in order to read the data in JavaScript.
 * const dbuf = new Float32Buffer([1.0, 2.0, 3.0]);
 * assert(dbuf.buffer instanceof DeviceMemory);
 *
 * // Override allocate function to create `ManagedMemory` instances.
 * setDefaultAllocator((byteLength) => new ManagedMemory(byteLength));
 *
 * // Now the allocator uses the supplied function to create
 * // `ManagedMemory` instances. This kind of memory can be accessed
 * // by both the CPU and GPU, because the CUDA driver automatically
 * // migrates the data from the CPU <-> GPU as required.
 * const mbuf = new Float32Buffer([1.0, 2.0, 3.0]);
 * assert(mbuf.buffer instanceof ManagedMemory);
 * ```
 *
 * @param allocate Function to use for device {@link Memory `Memory`} allocations.
 */
export function setDefaultAllocator(allocate?: null|((byteLength: number) => Memory)) {
  if (allocate === undefined || allocate === null) {
    // If allocate is null or undefined, reset to the default
    allocateMemory = allocateDeviceMemory;
  } else if (typeof allocate !== 'function') {
    throw new TypeError('setDefaultAllocator requires an `allocate` function');
  } else {
    // Validate the user-provided function returns something we expect.
    const mem = allocate(8);
    if (!isMemoryLike(mem) || (mem.byteLength !== 8)) {
      throw new TypeError(
        'setDefaultAllocator requires the `allocate` function to return Memory instances');
    }
    allocateMemory = allocate;
  }
}

/**
 * @summary A base class for typed arrays of values in owned or managed by CUDA.
 */
export abstract class MemoryView<T extends TypedArray|BigIntArray = any> implements
  ArrayBufferView {
  public static readonly BYTES_PER_ELEMENT: number;

  /**
   * @summary The size in bytes of each element in the MemoryView.
   */
  public readonly BYTES_PER_ELEMENT!: number;

  /**
   * @summary The {@link Memory `Memory`} instance referenced by the MemoryView.
   */
  public readonly buffer!: Memory;

  /**
   * @summary The offset in bytes of the MemoryView.
   */
  public readonly byteOffset!: number;

  /**
   * @summary The length in bytes of the MemoryView.
   */
  public readonly byteLength!: number;

  /**
   * @summary The length of the MemoryView.
   */
  public readonly length!: number;

  [index: number]: T[0];

  /**
   * @summary The constructor of the MemoryView's corresponding JS TypedArray.
   */
  public readonly TypedArray!: TypedArrayConstructor<T>;

  /**
   * @ignore
   * @summary The constructor function for the MemoryView type.
   */
  public readonly[Symbol.species]!: MemoryViewConstructor<T>;

  constructor(length?: number);
  constructor(arrayOrArrayBuffer: Iterable<T[0]>|ArrayLike<T[0]>|MemoryData);
  constructor(buffer: ArrayLike<T[0]>|MemoryData, byteOffset: number, length?: number);
  constructor() {
    // eslint-disable-next-line prefer-const, prefer-rest-params
    let [buffer, byteOffset, length] = arguments;
    Object.assign(this, toMemory(buffer, this.TypedArray));
    switch (arguments.length) {
      // @ts-ignore
      case 3:
        this.length = length = max(+length, 0) || 0;
        this.byteLength      = length * this.BYTES_PER_ELEMENT;
      // @ts-ignore
      // eslint-disable-next-line no-fallthrough
      case 2: this.byteOffset = max(+byteOffset, 0) || 0; break;
    }
  }

  /**
   * Copies data from a region of a source {@link MemoryView}, {@link TypedArray}, or Array to a
   * region in this {@link MemoryView}, even if the source region overlaps with this {@link
   * MemoryView}.
   * @param source The {@link MemoryView}, {@link TypedArray}, or Array to copy
   *   from.
   * @param sourceStart The offset in `source` at which to begin copying. <b>Default:</b> `0`.
   * @param targetStart The offset in `this` from which to begin writing. <b>Default:</b> `0`.
   * @param targetEnd The offset in `this` at which to stop writing (not inclusive).
   *   <b>Default:</b> `this.length - targetStart`.
   * @returns `this`
   */
  public copyFrom(source: MemoryData|Iterable<number|bigint>|ArrayLike<number|bigint>,
                  sourceStart = 0,
                  targetStart = 0,
                  targetEnd   = this.length) {
    this.subarray(targetStart, targetEnd)
      .set(toHDView(source, this.TypedArray).subarray(sourceStart));
    return this;
  }

  /**
   * Copies data from a region of this {@link MemoryView} to a region in a target {@link
   * MemoryView}, {@link TypedArray}, or Array, even if the target region overlaps with this {@link
   * MemoryView}.
   * @param target The {@link MemoryView}, {@link TypedArray}, or Array to copy
   *   into.
   * @param targetStart The offset in `target` at which to begin writing. <b>Default:</b> `0`.
   * @param sourceStart The offset in `this` from which to begin copying. <b>Default:</b> `0`.
   * @param sourceEnd The offset in `this` at which to stop copying (not inclusive). <b>Default:</b>
   *   <b>Default:</b> `this.length - sourceStart`.
   * @returns `this`
   */
  public copyInto(target: MemoryData|Array<any>,
                  targetStart = 0,
                  sourceStart = 0,
                  sourceEnd   = this.length) {
    if (!target) {
      throw new TypeError(
        `${this[Symbol.toStringTag]}.copyInto argument "target" cannot be null or undefined`);
    }
    const source = this.subarray(...clampRange(this.length, sourceStart, sourceEnd));
    if (target instanceof MemoryView || isMemoryLike(target)) {
      toMemoryView(target, this.TypedArray).set(source, targetStart);
    } else if (isArrayBufferLike(target) || isArrayBufferView(target)) {
      // If target is a ArrayBuffer or ArrayBufferView, copy from device to host via cudaMemcpy
      const destination = toHDView(target, this.TypedArray).subarray(targetStart);
      cudaMemcpy(destination, source, min(destination.byteLength, source.byteLength));
    } else if (Array.isArray(target)) {
      // If target is an Array, copy the data from device to host and splice the values into place
      target.splice(targetStart, 0, ...source.toArray());
    } else {
      throw new TypeError(`${this[Symbol.toStringTag]}.copyInto argument "target" invalid type`);
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
  public set(array: MemoryData|ArrayLike<number>|ArrayLike<bigint>, start?: number) {
    const [begin, end] = clampRange(this.length, start);
    const source       = toHDView(array, this.TypedArray);
    const length       = min((end - begin) * this.BYTES_PER_ELEMENT, source.byteLength);
    // const length       = min(end * this.BYTES_PER_ELEMENT, source.byteLength);
    cudaMemcpy(this.subarray(begin), source, length);
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
    [start, end] = clampRange(this.length, start, end);
    this.set(new this.TypedArray(end - start).fill(<never>value), start);
    return this;
  }

  /**
   * Returns a section of an array.
   * @param start The beginning of the specified portion of the array.
   * @param end The end of the specified portion of the array. This is exclusive of the element at
   *   the index 'end'.
   */
  public slice(start?: number, end?: number) {
    [start, end] = clampRange(this.length, start, end);
    return new this[Symbol.species](
      this.buffer.slice(this.byteOffset + (start * this.BYTES_PER_ELEMENT),
                        this.byteOffset + (end * this.BYTES_PER_ELEMENT)));
  }

  /**
   * Creates a new MemoryView view over the underlying Memory of this array,
   * referencing the elements at begin, inclusive, up to end, exclusive.
   * @param begin The index of the beginning of the array.
   * @param end The index of the end of the array.
   */
  public subarray(begin?: number, end?: number) {
    [begin, end] = clampRange(this.length, begin, end);
    return new this[Symbol.species](
      this.buffer, this.byteOffset + (begin * this.BYTES_PER_ELEMENT), end - begin);
  }

  /** @ignore */
  public get[Symbol.toStringTag]() { return this.constructor.name; }

  /** @ignore */
  public[Symbol.for('nodejs.util.inspect.custom')]() { return this.toString(); }

  /** @ignore */
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
   * @summary Create an IpcHandle for the underlying CUDA device memory.
   */
  public getIpcHandle() {
    if (!(this.buffer instanceof DeviceMemory)) {
      throw new Error(`${this[Symbol.toStringTag]}'s buffer must be an instance of DeviceMemory`);
    }
    return new IpcHandle(this.buffer, this.byteOffset);
  }
}

Object.setPrototypeOf(MemoryView.prototype, new Proxy({}, {
                        get(target: any, p: any, receiver: any) {
                          let i: number = p;
                          switch (typeof p) {
                            // @ts-ignore
                            case 'string':
                              if (isNaN(i = +p)) { break; }
                            // eslint-disable-next-line no-fallthrough
                            case 'number':
                              if (i > -1 && i < receiver.length) {
                                const {byteOffset, BYTES_PER_ELEMENT, E} = receiver;
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
                              if (isNaN(i = +p)) { break; }
                            // eslint-disable-next-line no-fallthrough
                            case 'number':
                              if (i > -1 && i < receiver.length) {
                                const {byteOffset, BYTES_PER_ELEMENT, E} = receiver;
                                // eslint-disable-next-line @typescript-eslint/restrict-plus-operands
                                receiver.byteOffset = byteOffset + i * BYTES_PER_ELEMENT;
                                E[0]                = value;
                                cudaMemcpy(receiver, E, BYTES_PER_ELEMENT);
                                receiver.byteOffset = byteOffset;
                                return true;
                              }
                          }
                          return Reflect.set(target, p, value, receiver);
                        }
                      }));

/** @ignore */ (<any>MemoryView.prototype).buffer            = new DeviceMemory(0);
/** @ignore */ (<any>MemoryView.prototype).length            = 0;
/** @ignore */ (<any>MemoryView.prototype).byteOffset        = 0;
/** @ignore */ (<any>MemoryView.prototype).byteLength        = 0;
/** @ignore */ (<any>MemoryView.prototype)[Symbol.species]   = MemoryView;
/** @ignore */ (<any>MemoryView.prototype).TypedArray        = Uint8ClampedArray;
/** @ignore */ (<any>MemoryView.prototype).E                 = new Uint8ClampedArray(8);
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

[{0: Int8Buffer, 1: Int8Array},
 {0: Int16Buffer, 1: Int16Array},
 {0: Int32Buffer, 1: Int32Array},
 {0: Uint8Buffer, 1: Uint8Array},
 {0: Uint8ClampedBuffer, 1: Uint8ClampedArray},
 {0: Uint16Buffer, 1: Uint16Array},
 {0: Uint32Buffer, 1: Uint32Array},
 {0: Float32Buffer, 1: Float32Array},
 {0: Float64Buffer, 1: Float64Array},
 {0: Int64Buffer, 1: BigInt64Array},
 {0: Uint64Buffer, 1: BigUint64Array},
].forEach(({0: MemoryViewCtor, 1: TypedArrayCtor}) => {
  (<any>MemoryViewCtor.prototype).TypedArray        = TypedArrayCtor;
  (<any>MemoryViewCtor.prototype)[Symbol.species]   = MemoryViewCtor;
  (<any>MemoryViewCtor).BYTES_PER_ELEMENT           = TypedArrayCtor.BYTES_PER_ELEMENT;
  (<any>MemoryViewCtor.prototype).BYTES_PER_ELEMENT = TypedArrayCtor.BYTES_PER_ELEMENT;
  (<any>MemoryViewCtor.prototype).E = new TypedArrayCtor((<any>MemoryView.prototype).E.buffer);
});

/** @internal */
function toMemory<T extends TypedArray|BigIntArray>(
  source: number|Iterable<T[0]>|ArrayLike<T[0]>|MemoryData, TypedArray: TypedArrayConstructor<T>) {
  let byteOffset = 0;
  let byteLength = 0;
  let buffer: Memory;
  if (isNumber(source)) {
    byteLength = source * TypedArray.BYTES_PER_ELEMENT;
    buffer     = allocateMemory(source * TypedArray.BYTES_PER_ELEMENT);
    // initialize with new allocated memory with zeroes
    cudaMemset(buffer, 0, byteLength);
  } else if (isMemoryLike(source)) {
    // If source is a device Memory instance, don't copy it
    buffer     = source;
    byteLength = source.byteLength;
  } else if ((source instanceof MemoryView)  //
             || isArrayBufferLike(source)    //
             || isArrayBufferView(source)) {
    // If source is a host ArrayBuffer[View] or MemoryView, make a device copy
    byteLength = source.byteLength;
    buffer     = allocateMemory(byteLength);
    cudaMemcpy(buffer, source, byteLength);
  } else if (isIterable(source) || isArrayLike(source)) {
    // If source is an Iterable or JavaScript Array, construct a TypedArray from the values
    const array = TypedArray.from(source, TypedArray.name.includes('Big') ? BigInt : Number);
    byteLength  = array.byteLength;
    buffer      = allocateMemory(byteLength);
    cudaMemcpy(buffer, array.buffer, byteLength);
  } else if (isObject(source) && ('buffer' in source)) {
    ({buffer, byteOffset, byteLength} = toMemoryView(source, TypedArray));
  } else {
    byteOffset = 0;
    byteLength = 0;
    buffer     = allocateMemory(0);
  }
  return {buffer, byteLength, byteOffset, length: byteLength / TypedArray.BYTES_PER_ELEMENT};
}

/**
 * @internal
 *
 * @summary Construct and return a MemoryView corresponding to the given TypedArray.
 * If necessary, copy data from the source host CPU arrays or buffers to device Memory.
 *
 * @note If the source is already a Memory or MemoryView, this function will create a
 * new MemoryView of the requested type without copying the underlying device Memory.
 *
 * @param source The source data from which to construct a GPU MemoryView.
 * @param TypedArray The MemoryView corresponding to the requested TypedArray.
 * @returns A MemoryView corresponding to the given TypedArray type.
 */
function toMemoryView<T extends TypedArray|BigIntArray>(
  source: Iterable<T[0]>|ArrayLike<T[0]>|MemoryData, TypedArray: TypedArrayConstructor<T>) {
  if (source instanceof MemoryView && source.TypedArray === TypedArray) {
    // If source is already the requested type, return it
    return source as MemoryViewOf<T>;
  }

  let buffer     = source as MemoryData;
  let byteOffset = 0, byteLength: number|undefined;
  if ('byteOffset' in source) { ({byteOffset} = source); }
  if ('byteLength' in source) { ({byteLength} = source); }
  while (('buffer' in buffer) && (buffer['buffer'] !== buffer)) {  //
    buffer = buffer['buffer'];
  }

  buffer = ((source: MemoryData) => {
    switch (TypedArray.name) {
      case 'Int8Array': return new Int8Buffer(source);
      case 'Int16Array': return new Int16Buffer(source);
      case 'Int32Array': return new Int32Buffer(source);
      case 'Uint8Array': return new Uint8Buffer(source);
      case 'Uint8ClampedArray': return new Uint8ClampedBuffer(source);
      case 'Uint16Array': return new Uint16Buffer(source);
      case 'Uint32Array': return new Uint32Buffer(source);
      case 'Float32Array': return new Float32Buffer(source);
      case 'Float64Array': return new Float64Buffer(source);
      case 'BigInt64Array': return new Int64Buffer(source);
      case 'BigUint64Array': return new Uint64Buffer(source);
    }
    throw new Error('Unknown dtype');
  })(source as MemoryData);

  if (byteLength !== undefined) {
    (<any>buffer).byteOffset = byteOffset;
    (<any>buffer).byteLength = byteLength;
    (<any>buffer).length     = byteLength / TypedArray.BYTES_PER_ELEMENT;
  }

  return buffer as MemoryViewOf<T>;
}

/**
 * @internal
 *
 * @summary Construct a host TypedArray or device `MemoryView` based on the location of the input
 * `source` data.
 *
 * * If the source data is already a `Memory` or `MemoryView`, construct and return a `MemoryView`
 *   corresponding to the desired TypedArray.
 * * If the source data is already an `ArrayBuffer` or `ArrayBufferView`, construct and return a
 *   TypedArray of the desired type.
 * * If the source data is a JavaScript Iterable or Array, construct and return a TypedArray of the
 *   desired type by enumerating the source values.
 * * If the source data is a JavaScript Object with a "buffer" member, construct either a host
 *   TypedArray or device `MemoryView` depending on the location of the underling buffer.
 *
 * @param source The source data from which to construct a CPU TypedArray or GPU `MemoryView`.
 * @param TypedArray The TypedArray to return (if source is on the host) or its corresponding
 *   `MemoryView` (if source is on the device).
 * @returns A TypedArray or `MemoryView` corresponding to the desired TypedArray type.
 */
function toHDView<T extends TypedArray|BigIntArray>(
  source: Iterable<number|bigint>|ArrayLike<number|bigint>|MemoryData,
  TypedArray: TypedArrayConstructor<T>): T|MemoryViewOf<T> {
  if (source instanceof MemoryView) {
    return (source.TypedArray === TypedArray)
             // If source is already the desired type, return it
             ? source as MemoryViewOf<T>
             // If source is another type of MemoryView, wrap in the desired type
             : toMemoryView(source, TypedArray);
  } else if (isMemoryLike(source)) {
    // If source is MemoryLike, wrap it in a MemoryView of the desired type
    return toMemoryView(source, TypedArray);
  } else if (isArrayBufferLike(source)) {
    // If source is an ArrayBuffer or SharedArrayBuffer, wrap in a TypedArray of the desired type
    return new TypedArray(source);
  } else if (isArrayBufferView(source)) {
    // If source is already an ArrayBufferView of the desired type, return it
    if (source.constructor === TypedArray) { return source as T; }
    // If source is an ArrayyBufferView of another kind, return a TypedArray of the desired type
    return new TypedArray(source.buffer,
                          source.byteOffset,  //
                          source.byteLength / TypedArray.BYTES_PER_ELEMENT);
  } else if (isIterable(source) || isArrayLike(source)) {
    // If source is an Iterable or Array, construct a TypedArray of the desired type
    return TypedArray.from(source, TypedArray.name.includes('Big') ? BigInt : Number);
  }
  if (isObject(source) && ('buffer' in source)) {
    // If source is a JS object with a 'buffer' key, recurse down to wrap either as a
    // MemoryView or TypedArray based on whether buffer is a Memory or ArrayBuffer instance.
    let buffer     = source as MemoryData;
    let byteOffset = 0, byteLength: number|undefined;
    if ('byteOffset' in source) { ({byteOffset} = source); }
    if ('byteLength' in source) { ({byteLength} = source); }
    while (('buffer' in buffer) && (buffer['buffer'] !== buffer)) {  //
      buffer = buffer['buffer'];
    }
    buffer = toHDView(buffer, TypedArray);
    if (byteLength !== undefined) {
      (<any>buffer).byteOffset = byteOffset;
      (<any>buffer).byteLength = byteLength;
      (<any>buffer).length     = byteLength / TypedArray.BYTES_PER_ELEMENT;
    }
    return buffer as MemoryViewOf<T>| T;
  }
  throw new TypeError(
    'asMemoryData() received invalid "source". Expected a MemoryData, Iterable, Array, or Object with a {"buffer"} `source`.');
}
