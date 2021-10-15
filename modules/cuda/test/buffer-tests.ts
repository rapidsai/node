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

import {
  BigIntArray,
  Float32Buffer,
  Float64Buffer,
  Int16Buffer,
  Int32Buffer,
  Int64Buffer,
  Int8Buffer,
  MemoryViewConstructor,
  setDefaultAllocator,
  TypedArray,
  Uint16Buffer,
  Uint32Buffer,
  Uint64Buffer,
  Uint8Buffer,
} from '@rapidsai/cuda';

describe.each(<[MemoryViewConstructor<TypedArray|BigIntArray>, typeof Number | typeof BigInt][]>[
  [Int8Buffer, Number],
  [Int16Buffer, Number],
  [Int32Buffer, Number],
  [Int64Buffer, BigInt],
  [Uint8Buffer, Number],
  [Uint16Buffer, Number],
  [Uint32Buffer, Number],
  [Uint64Buffer, BigInt],
  [Float32Buffer, Number],
  [Float64Buffer, Number],
])(`%s`,
   <T extends TypedArray|BigIntArray, TValue extends typeof Number|typeof BigInt>(
     BufferCtor: MemoryViewConstructor<T>, ValueCtor: TValue) => {
     const values = Array.from({length: 1024}, (_, i) => ValueCtor(i));
     const buffer = BufferCtor.TypedArray.from(values);

     beforeEach(() => setDefaultAllocator(null));

     test(`constructs ${BufferCtor.name} from a JavaScript Array via HtoD copy`, () => {
       const dbuf = new BufferCtor(values);
       expect(dbuf.toArray()).toEqual(buffer);
     });

     test(`constructs ${BufferCtor.name} from a JavaScript Iterable via HtoD copy`, () => {
       const dbuf = new BufferCtor(function*() { yield* values; }());
       expect(dbuf.toArray()).toEqual(buffer);
     });

     test(`constructs ${BufferCtor.name} from an ArrayBuffer via HtoD copy`, () => {
       const dbuf = new BufferCtor(buffer.buffer);
       expect(dbuf.toArray()).toEqual(buffer);
     });

     test(`constructs ${BufferCtor.name} from an ArrayBufferView via HtoD copy`, () => {
       const dbuf = new BufferCtor(buffer);
       expect(dbuf.toArray()).toEqual(buffer);
     });

     test(`constructs ${BufferCtor.name} from a device Memory instance zero-copy`, () => {
       const mem  = new BufferCtor(buffer).buffer;
       const dbuf = new BufferCtor(mem);
       expect(dbuf.toArray()).toEqual(buffer);
       expect(dbuf.buffer === mem).toBe(true);
       expect(dbuf.buffer.ptr).toEqual(mem.ptr);
     });

     test(`constructs ${BufferCtor.name} from a device MemoryView via DtoD copy`, () => {
       const dbuf = new BufferCtor(new BufferCtor(buffer));
       expect(dbuf.toArray()).toEqual(buffer);
     });

     test(`reads ${BufferCtor.name} values via subscript accessor`, () => {
       const dbuf = new BufferCtor(buffer);
       for (let i = -1; ++i < dbuf.length;) { expect(dbuf[i]).toEqual(buffer[i]); }
     });

     test(`writes ${BufferCtor.name} values via subscript accessor`, () => {
       const dbuf = new BufferCtor(buffer);
       const mult = <T[0]>buffer[buffer.length * .17 | 0];
       (() => {
         for (let i = -1, n = dbuf.length; ++i < n;) {  //
           dbuf[i] = (<T[0]>buffer[i]) * mult;
         }
       })();
       const results = [...buffer].map((i: T[0]) => i * mult);
       expect(dbuf.toArray()).toEqual(BufferCtor.TypedArray.from(results));
     });

     test(`slice copies the device memory`, () => {
       const dbuf = new BufferCtor(buffer);
       const copy = dbuf.slice();
       expect(copy.toArray()).toEqual(buffer);
       expect(dbuf.buffer.ptr).not.toEqual(copy.buffer.ptr);
     });

     test(`slice copies the device memory range`, () => {
       const start = 300, end = 700;
       const dbuf = new BufferCtor(buffer);
       const copy = dbuf.slice(start, end);
       expect(copy.toArray()).toEqual(buffer.slice(start, end));
       expect(dbuf.buffer.ptr).not.toEqual(copy.buffer.ptr);
       expect(copy.byteOffset).toEqual(buffer.slice(start, end).byteOffset);
       expect(copy.byteLength).toEqual(buffer.slice(start, end).byteLength);
     });

     test(`subarray does not copy the device memory`, () => {
       const dbuf = new BufferCtor(buffer);
       const span = dbuf.subarray();
       expect(span.toArray()).toEqual(buffer);
       expect(dbuf.buffer.ptr).toEqual(span.buffer.ptr);
     });

     test(`subarray does not copy the device memory range`, () => {
       const start = 300, end = 700;
       const dbuf = new BufferCtor(buffer);
       const span = dbuf.subarray(start, end);
       expect(span.toArray()).toEqual(buffer.subarray(start, end));
       expect(dbuf.buffer.ptr).toEqual(span.buffer.ptr);
       expect(span.byteOffset).toEqual(buffer.subarray(start, end).byteOffset);
       expect(span.byteLength).toEqual(buffer.subarray(start, end).byteLength);
     });

     test(`can copy from unregistered host memory`, () => {
       const source = buffer.slice();
       const target = new BufferCtor(source.length);
       target.copyFrom(source);
       expect(target.toArray()).toEqual(source);
     });

     test(`can copy into unregistered host memory`, () => {
       const source = new BufferCtor(buffer);
       const target = new BufferCtor.TypedArray(source.length);
       source.copyInto(target);
       expect(target).toEqual(buffer);
     });

     test(`can copy from device memory with offsets and lengths`, () => {
       const source = new BufferCtor(buffer);
       const target = new BufferCtor(buffer.length);
       // swap the halves
       target.copyFrom(source, 0, target.length / 2, target.length);
       target.copyFrom(source, source.length / 2, 0, target.length / 2);
       expect(target.toArray()).toEqual(new BufferCtor.TypedArray([
         ...buffer.subarray(buffer.length / 2),
         ...buffer.subarray(0, buffer.length / 2)
       ]));
     });

     test(`can copy into device memory with offsets and lengths`, () => {
       const source = new BufferCtor(buffer);
       const target = new BufferCtor(buffer.length);
       // swap the halves
       source.copyInto(target, 0, source.length / 2, source.length);
       source.copyInto(target, target.length / 2, 0, source.length / 2);
       expect(target.toArray()).toEqual(new BufferCtor.TypedArray([
         ...buffer.subarray(buffer.length / 2),  //
         ...buffer.subarray(0, buffer.length / 2)
       ]));
     });

     test(`can copy from unregistered host memory with offsets and lengths`, () => {
       const source = buffer.slice();
       const target = new BufferCtor(buffer.length);
       // swap the halves
       target.copyFrom(source, 0, target.length / 2, target.length);
       target.copyFrom(source, source.length / 2, 0, target.length / 2);
       expect(target.toArray()).toEqual(new BufferCtor.TypedArray([
         ...buffer.subarray(buffer.length / 2),
         ...buffer.subarray(0, buffer.length / 2)
       ]));
     });

     test(`can copy into unregistered host memory with offsets and lengths`, () => {
       const source = new BufferCtor(buffer);
       const target = new BufferCtor.TypedArray(source.length);
       // swap the halves
       source.copyInto(target, 0, source.length / 2, source.length);
       source.copyInto(target, buffer.length / 2, 0, source.length / 2);
       expect(target).toEqual(new BufferCtor.TypedArray([
         ...buffer.subarray(buffer.length / 2),  //
         ...buffer.subarray(0, buffer.length / 2)
       ]));
     });
   });
