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

import {
  Float32Buffer,
  Float64Buffer,
  Int16Buffer,
  Int32Buffer,
  Int64Buffer,
  Int8Buffer,
  Uint16Buffer,
  Uint32Buffer,
  Uint64Buffer,
  Uint8Buffer,
} from '@nvidia/cuda';

describe.each([
  Int8Buffer,
  Int16Buffer,
  Int32Buffer,
  Int64Buffer,
  Uint8Buffer,
  Uint16Buffer,
  Uint32Buffer,
  Uint64Buffer,
  Float32Buffer,
  Float64Buffer,
])(`%s`, (Buffer) => {
  const testNums        = Array.from({length: 1024}, (_, i) => i);
  const testValues: any = (() => {
    switch (Buffer) {
      case Int64Buffer:
      case Uint64Buffer: return testNums.map((i) => BigInt(i));
      default: return testNums;
    }
  })();

  const testBuffer: any = Buffer.TypedArray.from(testValues);

  test(`constructs ${Buffer.name} from a JS Array`, () => {
    const dbuf = new Buffer(testValues);
    expect(dbuf.toArray()).toEqual(testBuffer);
  });

  test(`constructs ${Buffer.name} from a Typed Array`, () => {
    const dbuf = new Buffer(testBuffer);
    expect(dbuf.toArray()).toEqual(testBuffer);
  });

  test(`reads ${Buffer.name} values via subscript accessor`, () => {
    const dbuf = new Buffer(testBuffer);
    for (let i = -1; ++i < dbuf.length;) { expect(dbuf [i]).toEqual(testBuffer [i]); }
  });

  test(`writes ${Buffer.name} values via subscript accessor`, () => {
    const dbuf = new Buffer(testBuffer);
    const mult = testBuffer [testBuffer.length * .25 | 0];
    for (let i = -1; ++i < dbuf.length;) { dbuf [i] = testBuffer [i] * mult; }
    expect(dbuf.toArray()).toEqual(testBuffer.map((i: any) => i * mult));
  });

  test(`slice copies the device memory`, () => {
    const dbuf = new Buffer(testBuffer);
    const copy = dbuf.slice();
    expect(copy.toArray()).toEqual(testBuffer);
    expect(dbuf.buffer.ptr).not.toEqual(copy.buffer.ptr);
  });

  test(`slice copies the device memory range`, () => {
    const start = 300, end = 700;
    const dbuf = new Buffer(testBuffer);
    const copy = dbuf.slice(start, end);
    expect(copy.toArray()).toEqual(testBuffer.slice(start, end));
    expect(dbuf.buffer.ptr).not.toEqual(copy.buffer.ptr);
    expect(copy.byteOffset).toEqual(testBuffer.slice(start, end).byteOffset);
    expect(copy.byteLength).toEqual(testBuffer.slice(start, end).byteLength);
  });

  test(`subarray does not copy the device memory`, () => {
    const dbuf = new Buffer(testBuffer);
    const span = dbuf.subarray();
    expect(span.toArray()).toEqual(testBuffer);
    expect(dbuf.buffer.ptr).toEqual(span.buffer.ptr);
  });

  test(`subarray does not copy the device memory range`, () => {
    const start = 300, end = 700;
    const dbuf = new Buffer(testBuffer);
    const span = dbuf.subarray(start, end);
    expect(span.toArray()).toEqual(testBuffer.subarray(start, end));
    expect(dbuf.buffer.ptr).toEqual(span.buffer.ptr);
    expect(span.byteOffset).toEqual(testBuffer.subarray(start, end).byteOffset);
    expect(span.byteLength).toEqual(testBuffer.subarray(start, end).byteLength);
  });

  test(`can copy from unregistered host memory`, () => {
    const dbuf = new Buffer(testBuffer.length);
    dbuf.copyFrom(testBuffer);
    expect(dbuf.toArray()).toEqual(testBuffer);
  });

  test(`can copy into unregistered host memory`, () => {
    const dbuf = new Buffer(testBuffer);
    const hbuf = new Buffer.TypedArray(dbuf.length);
    dbuf.copyInto(hbuf);
    expect(hbuf).toEqual(testBuffer);
  });
});
