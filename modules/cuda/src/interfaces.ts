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

/** @ignore */
export type FloatArray = Float32Array|Float64Array;

/** @ignore */
export type IntArray = Int8Array|Int16Array|Int32Array;

/** @ignore */
export type UintArray = Uint8Array|Uint16Array|Uint32Array|Uint8ClampedArray;

/** @ignore */
export type BigIntArray = BigInt64Array|BigUint64Array;

/** @ignore */
export type TypedArray = FloatArray|IntArray|UintArray;

/** @ignore */
export type TypedArrayConstructor<T extends TypedArray|BigIntArray> = {
  readonly BYTES_PER_ELEMENT: number; new (length?: number): T; new (values: Iterable<T[0]>): T;
  new (buffer: ArrayBufferLike, byteOffset?: number, length?: number): T;
  from(arrayLike: Iterable<T[0]>|ArrayLike<T[0]>): T;
  from(
    arrayLike: Iterable<T[0]>|ArrayLike<T[0]>, mapfn: (v: T[0], k: number) => T[0], thisArg?: any):
    T;
};

/** @ignore */
export type MemoryData = TypedArray|BigIntArray|ArrayBufferView|ArrayBufferLike  //
  |(import('./addon').DeviceMemory)                                              //
  |(import('./addon').PinnedMemory)                                              //
  |(import('./addon').ManagedMemory)                                             //
  |(import('./addon').IpcMemory)                                                 //
  |(import('./addon').MappedGLMemory);
