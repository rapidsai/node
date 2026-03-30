// Copyright (c) 2026, NVIDIA CORPORATION.
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
  setDefaultAllocator,
  Uint16Buffer,
  Uint32Buffer,
  Uint64Buffer,
  Uint8Buffer,
  Uint8ClampedBuffer
} from '@rapidsai/cuda';
import {
  Bool8,
  Column,
  Float32,
  Float32Series,
  Float64,
  Float64Series,
  Int16,
  Int16Series,
  Int32,
  Int32Series,
  Int64,
  Int64Series,
  Int8,
  Int8Series,
  Series,
  Uint16,
  Uint16Series,
  Uint32,
  Uint32Series,
  Uint64,
  Uint64Series,
  Uint8,
  Uint8Series,
  Utf8String
} from '@rapidsai/cudf';
import {DeviceBuffer} from '@rapidsai/rmm';
import * as arrow from 'apache-arrow';

setDefaultAllocator((byteLength: number) => new DeviceBuffer(byteLength));

test('Series initialization with type inference', () => {
  const a = Series.new([0, 1, 2, null]);
  const b = Series.new(['foo', 'bar', 'test', null]);
  const c = Series.new([0n, 1n, 2n, null]);
  const d = Series.new([true, false, true, null]);

  expect(a.type).toBeInstanceOf(Float64);
  expect(b.type).toBeInstanceOf(Utf8String);
  expect(c.type).toBeInstanceOf(Int64);
  expect(d.type).toBeInstanceOf(Bool8);
});

test('Series initialization with Series child', () => {
  const utf8Col    = Series.new({type: new Uint8, data: new Uint8Buffer(Buffer.from('hello'))});
  const offsetsCol = Series.new({type: new Int32, data: new Int32Buffer([0, utf8Col.length])});
  const stringsCol = Series.new({
    type: new Utf8String(),
    data: utf8Col.data,
    children: [offsetsCol],
  });

  expect(stringsCol.type).toBeInstanceOf(Utf8String);
  expect(stringsCol.numChildren).toBe(1);
  expect(stringsCol.nullCount).toBe(0);
  expect(stringsCol.getValue(0)).toBe('hello');
  expect(stringsCol.offsets.length).toBe(offsetsCol.length);
  expect(stringsCol.offsets.type).toBeInstanceOf(Int32);
  expect(stringsCol.data.length).toBe(utf8Col.length);
  expect(stringsCol.data.type).toBeInstanceOf(Uint8);
});

test('Series initialization with Column child', () => {
  const utf8Col    = Series.new({type: new Uint8, data: new Uint8Buffer(Buffer.from('hello'))});
  const offsetsCol = Series.new({type: new Int32, data: new Int32Buffer([0, utf8Col.length])});
  const stringsCol = Series.new({
    type: new Utf8String(),
    data: utf8Col.data,
    children: [offsetsCol._col],
  });

  expect(stringsCol.type).toBeInstanceOf(Utf8String);
  expect(stringsCol.numChildren).toBe(1);
  expect(stringsCol.nullCount).toBe(0);
  expect(stringsCol.getValue(0)).toBe('hello');
  expect(stringsCol.offsets.length).toBe(offsetsCol.length);
  expect(stringsCol.offsets.type).toBeInstanceOf(Int32);
  expect(stringsCol.data.length).toBe(utf8Col.length);
  expect(stringsCol.data.type).toBeInstanceOf(Uint8);
});

test('Series initialization with Date', () => {
  const dateTime = new Date('May 13, 2021 16:38:30:100 GMT+00:00');
  const s        = Series.new([dateTime]);
  const val      = s.getValue(0);

  expect(val?.getUTCFullYear()).toBe(2021);
  expect(val?.getUTCMonth()).toBe(4);
  expect(val?.getUTCDate()).toBe(13);
  expect(val?.getUTCHours()).toBe(16);
  expect(val?.getUTCMinutes()).toBe(38);
  expect(val?.getUTCSeconds()).toBe(30);
  expect(val?.getUTCMilliseconds()).toBe(100);
});

describe('toArrow()', () => {
  test('converts Uint8 Series to Uint8Vector', () => {
    const uint8Col = Series.new({type: new Uint8, data: new Uint8Buffer(Buffer.from('hello'))});
    const uint8Vec = uint8Col.toArrow();
    expect(uint8Vec).toBeInstanceOf(arrow.Vector);
    expect([...uint8Vec]).toEqual([...Buffer.from('hello')]);
  });
  test('converts String Series to Utf8Vector', () => {
    const utf8Col    = Series.new({type: new Uint8, data: new Uint8Buffer(Buffer.from('hello'))});
    const offsetsCol = Series.new({type: new Int32, data: new Int32Buffer([0, utf8Col.length])});
    const stringsCol = Series.new({
      type: new Utf8String(),
      data: utf8Col.data,
      children: [offsetsCol],
    });
    const utf8Vec    = stringsCol.toArrow();
    expect(utf8Vec).toBeInstanceOf(arrow.Vector);
    expect([...utf8Vec]).toEqual(['hello']);
  });
});

describe.each([
  new Int32,
  new Float32,
  new Float64,
])('Series.sequence({type=%p,, ...})', (typ) => {
  test('no step', () => {
    const col = Series.sequence({type: typ, size: 10, init: 0});
    expect([...col]).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  });
  test('step=1', () => {
    const col = Series.sequence({type: typ, size: 10, step: 1, init: 0});
    expect([...col]).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  });
  test('step=2', () => {
    const col = Series.sequence({type: typ, size: 10, step: 2, init: 0});
    expect([...col]).toEqual([0, 2, 4, 6, 8, 10, 12, 14, 16, 18]);
  });
});

describe.each([
  [Int8Series.name, Int8Series, Int8, Int8Array, Int8Buffer],
  [Int16Series.name, Int16Series, Int16, Int16Array, Int16Buffer],
  [Int32Series.name, Int32Series, Int32, Int32Array, Int32Buffer],
  [Int64Series.name, Int64Series, Int64, BigInt64Array, Int64Buffer],
  [Uint8Series.name, Uint8Series, Uint8, Uint8Array, Uint8Buffer],
  [Uint16Series.name, Uint16Series, Uint16, Uint16Array, Uint16Buffer],
  [Uint32Series.name, Uint32Series, Uint32, Uint32Array, Uint32Buffer],
  [Uint64Series.name, Uint64Series, Uint64, BigUint64Array, Uint64Buffer],
  [Float32Series.name, Float32Series, Float32, Float32Array, Float32Buffer],
  [Float64Series.name, Float64Series, Float64, Float64Array, Float64Buffer],
  [Uint8Series.name, Uint8Series, Uint8, Uint8ClampedArray, Uint8ClampedBuffer],
])(`%s initialization`,
   (_: string, SeriesType: any, DType: any, ArrayType: any, BufferType: any) => {
     const isBigInt = DType === Int64 || DType === Uint64;
     const values   = [1, 2, 3, 4, 5, 6].map(x => isBigInt ? BigInt(x) : x);
     const nulls    = values.slice() as any[];
     nulls[2]       = null;
     nulls[4]       = null;

     const cases = [
       [ArrayType.name, ArrayType],
       [BufferType.name, BufferType],
     ];

     test.each(cases)(`From %s`, (_: string, ArrayType: any) => {
       const v = new ArrayType(values);
       const s = Series.new(v);

       expect(s).toBeInstanceOf(SeriesType);
       expect(s.type).toBeInstanceOf(DType);
       expect(s.length).toBe(v.length);
       expect(s.offset).toBe((v.buffer instanceof DeviceBuffer) ? v.byteOffset / v.BYTES_PER_ELEMENT
                                                                : 0);
       expect(s.nullCount).toBe(0);
       expect(s.hasNulls).toBe(false);
       expect(s.nullable).toBe(false);
     });

     test.each(cases)(`From %s (sliced)`, (_: string, ArrayType: any) => {
       const v = new ArrayType(values).subarray(3);
       const s = Series.new(v);

       expect(s).toBeInstanceOf(SeriesType);
       expect(s.type).toBeInstanceOf(DType);
       expect(s.length).toBe(v.length);
       expect(s.offset).toBe((v.buffer instanceof DeviceBuffer) ? v.byteOffset / v.BYTES_PER_ELEMENT
                                                                : 0);
       expect(s.nullCount).toBe(0);
       expect(s.hasNulls).toBe(false);
       expect(s.nullable).toBe(false);
     });

     test.each(cases)(`From ColumnProps with data=%s (no nulls)`, (_: string, ArrayType: any) => {
       const v = new ArrayType(values);
       const s = Series.new({type: new DType, data: v});

       expect(s.type).toBeInstanceOf(DType);
       expect(s.length).toBe(v.length);
       expect(s.offset).toBe((v.buffer instanceof DeviceBuffer) ? v.byteOffset / v.BYTES_PER_ELEMENT
                                                                : 0);
       expect(s.nullCount).toBe(0);
       expect(s.hasNulls).toBe(false);
       expect(s.nullable).toBe(false);
     });

     test.each(cases)(`From ColumnProps with data=%s (with nulls)`, (_: string, ArrayType: any) => {
       const v = new ArrayType(values);
       const s = Series.new({type: new DType, data: v, nullMask: new Uint8Buffer([250])});

       expect(s.type).toBeInstanceOf(DType);
       expect(s.length).toBe(v.length);
       expect(s.offset).toBe((v.buffer instanceof DeviceBuffer) ? v.byteOffset / v.BYTES_PER_ELEMENT
                                                                : 0);
       expect(s.nullCount).toBe(2);
       expect(s.hasNulls).toBe(true);
       expect(s.nullable).toBe(true);
     });

     test.each(cases)(`From Column with data=%s`, (_: string, ArrayType: any) => {
       const v = new ArrayType(values);
       const s = Series.new(new Column({type: new DType, data: v}));

       expect(s.type).toBeInstanceOf(DType);
       expect(s.length).toBe(v.length);
       expect(s.offset).toBe((v.buffer instanceof DeviceBuffer) ? v.byteOffset / v.BYTES_PER_ELEMENT
                                                                : 0);
       expect(s.nullCount).toBe(0);
       expect(s.hasNulls).toBe(false);
       expect(s.nullable).toBe(false);
     });

     test(`From Array of mixed values`, () => {
       const v = nulls.slice();
       const s = Series.new({type: new DType, data: v});

       expect(s.type).toBeInstanceOf(DType);
       expect(s.length).toBe(v.length);
       expect(s.offset).toBe(0);
       expect(s.nullCount).toBe(2);
       expect(s.hasNulls).toBe(true);
       expect(s.nullable).toBe(true);
     });
   });
