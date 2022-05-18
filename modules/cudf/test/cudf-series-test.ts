// Copyright (c) 2020-2022, NVIDIA CORPORATION.
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
  StringSeries,
  TimestampDay,
  TimestampMicrosecond,
  TimestampMillisecond,
  TimestampNanosecond,
  TimestampSecond,
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
import {CudaMemoryResource, DeviceBuffer} from '@rapidsai/rmm';
import {Uint8Vector, Utf8Vector} from 'apache-arrow';
import {BoolVector} from 'apache-arrow';
import {promises} from 'fs';
import * as Path from 'path';

const mr = new CudaMemoryResource();

setDefaultAllocator((byteLength: number) => new DeviceBuffer(byteLength, mr));

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

test('test child(child_index), num_children', () => {
  const utf8Col    = Series.new({type: new Uint8, data: new Uint8Buffer(Buffer.from('hello'))});
  const offsetsCol = Series.new({type: new Int32, data: new Int32Buffer([0, utf8Col.length])});
  const stringsCol = Series.new({
    type: new Utf8String(),
    length: 1,
    nullMask: new Uint8Buffer([255]),
    children: [offsetsCol, utf8Col],
  });

  expect(stringsCol.type).toBeInstanceOf(Utf8String);
  expect(stringsCol.numChildren).toBe(2);
  expect(stringsCol.nullCount).toBe(0);
  expect(stringsCol.getValue(0)).toBe('hello');
  expect(stringsCol.offsets.length).toBe(offsetsCol.length);
  expect(stringsCol.offsets.type).toBeInstanceOf(Int32);
  expect(stringsCol.data.length).toBe(utf8Col.length);
  expect(stringsCol.data.type).toBeInstanceOf(Uint8);
});

test('test mixed series/column children', () => {
  const utf8Col    = Series.new({type: new Uint8, data: new Uint8Buffer(Buffer.from('hello'))});
  const offsetsCol = Series.new({type: new Int32, data: new Int32Buffer([0, utf8Col.length])});
  const stringsCol = Series.new({
    type: new Utf8String(),
    length: 1,
    nullMask: new Uint8Buffer([255]),
    children: [offsetsCol, utf8Col._col],
  });

  expect(stringsCol.type).toBeInstanceOf(Utf8String);
  expect(stringsCol.numChildren).toBe(2);
  expect(stringsCol.nullCount).toBe(0);
  expect(stringsCol.getValue(0)).toBe('hello');
  expect(stringsCol.offsets.length).toBe(offsetsCol.length);
  expect(stringsCol.offsets.type).toBeInstanceOf(Int32);
  expect(stringsCol.data.length).toBe(utf8Col.length);
  expect(stringsCol.data.type).toBeInstanceOf(Uint8);
});

test('Series.getValue', () => {
  const col = Series.new({type: new Int32, data: new Int32Buffer([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])});
  for (let i = 0; i < 10; i++) { expect(col.getValue(i)).toEqual(i); }
});

test('Series.setValue', () => {
  const col = Series.new({type: new Int32, data: new Int32Buffer([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])});
  col.setValue(2, 999);
  col.setValue(4, 999);
  col.setValue(5, 999);
  col.setValue(8, 999);
  expect([...col]).toEqual([0, 1, 999, 3, 999, 999, 6, 7, 999, 9]);
});

test('Series.setValues (series)', () => {
  const col     = Series.new({type: new Int32, data: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]});
  const values  = Series.new({type: new Int32, data: [200, 400, 500, 800]});
  const indices = Series.new({type: new Int32, data: [2, 4, 5, 8]});

  col.setValues(indices, values);

  expect([...col]).toEqual([0, 1, 200, 3, 400, 500, 6, 7, 800, 9]);
});

test('Series.setValues (scalar)', () => {
  const col     = Series.new({type: new Int32, data: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]});
  const indices = Series.new({type: new Int32, data: [2, 4, 5, 8]});

  col.setValues(indices, 999);

  expect([...col]).toEqual([0, 1, 999, 3, 999, 999, 6, 7, 999, 9]);
});

test('NumericSeries.cast', () => {
  const col = Series.new({type: new Int32, data: new Int32Buffer([0, 1, 2, 3, 4])});

  expect(col.cast(new Int64).type).toBeInstanceOf(Int64);
  expect(col.cast(new Float32).type).toBeInstanceOf(Float32);
  expect(col.cast(new Float64).type).toBeInstanceOf(Float64);

  const floatCol = Series.new({type: new Float32, data: new Float32Buffer([1.5, 2.8, 3.1, 4.2])});
  const result   = floatCol.cast(new Int32);

  expect([...result]).toEqual([1, 2, 3, 4]);
});

test('NumericSeries.concat', () => {
  const col         = Series.new({type: new Int32, data: new Int32Buffer([1, 2, 3, 4, 5])});
  const colToConcat = Series.new({type: new Int32, data: new Int32Buffer([6, 7, 8, 9, 10])});

  const result = col.concat(colToConcat);

  expect([...result]).toEqual([...col, ...colToConcat]);
});

test('NumericSeries.concat up-casts to common dtype', () => {
  const col         = Series.new([1, 2, 3, 4, 5]).cast(new Int32);
  const colToConcat = Series.new([6, 7, 8, 9, 10]);

  const result = col.concat(colToConcat);

  expect(result.type).toBeInstanceOf(Float64);
  expect([...result]).toEqual([...col, ...colToConcat]);
});

test('Series.copy fixed width', () => {
  const col = Series.new({type: new Int32, data: new Int32Buffer([1, 2, 3, 4, 5])});

  const result = col.copy();

  expect([...result]).toEqual([...col]);
});

test('Series.copy String', () => {
  const col = Series.new(['foo', 'bar', 'test', null]);

  const result = col.copy();

  expect([...result]).toEqual([...col]);
});

test('Series.gather', () => {
  const col = Series.new({type: new Int32, data: new Int32Buffer([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])});

  const selection = Series.new({type: new Int32, data: new Int32Buffer([2, 4, 5, 8])});

  const result = col.gather(selection);

  expect([...result]).toEqual([...selection]);
});

describe('Series.head', () => {
  const col = Series.new({type: new Int32, data: new Int32Buffer([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])});

  test('default n', () => { expect([...col.head()]).toEqual([0, 1, 2, 3, 4]); });

  test('invalid n', () => { expect(() => col.head(-1)).toThrowError(); });

  test('providing n', () => { expect([...col.head(8)]).toEqual([0, 1, 2, 3, 4, 5, 6, 7]); });

  test('n longer than length of series', () => {
    expect([...col.head(25)]).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  });
});

describe('Series.tail', () => {
  const col = Series.new({type: new Int32, data: new Int32Buffer([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])});

  test('default n', () => { expect([...col.tail()]).toEqual([5, 6, 7, 8, 9]); });

  test('invalid n', () => { expect(() => col.tail(-1)).toThrowError(); });

  test('providing n', () => { expect([...col.tail(8)]).toEqual([2, 3, 4, 5, 6, 7, 8, 9]); });

  test('n longer than length of series', () => {
    expect([...col.tail(25)]).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  });
});

describe('Series.nLargest', () => {
  const col =
    Series.new({type: new Int32, data: new Int32Buffer([9, 5, 0, 2, 1, 3, 4, 7, 6, 8, 0])});

  test('default n', () => { expect([...col.nLargest()]).toEqual([9, 8, 7, 6, 5]); });

  test('negative n', () => { expect([...col.nLargest(-1)]).toEqual([]); });

  test('providing n', () => { expect([...col.nLargest(8)]).toEqual([9, 8, 7, 6, 5, 4, 3, 2]); });

  test('n longer than length of series', () => {
    expect([...col.nLargest(25)]).toEqual([9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0]);
  });

  test('keep last duplicate option', () => {
    expect([...col.nLargest(25, 'last')]).toEqual([9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0]);

    expect([...col.nLargest(-5, 'last')]).toEqual([]);
  });

  test('keep none duplicate option throws',
       () => { expect(() => col.nLargest(25, 'none')).toThrow(); });
});

describe('Series.nSmallest', () => {
  const col =
    Series.new({type: new Int32, data: new Int32Buffer([9, 5, 0, 2, 1, 3, 4, 7, 6, 8, 0])});

  test('default n', () => { expect([...col.nSmallest()]).toEqual([0, 0, 1, 2, 3]); });

  test('negative n', () => { expect([...col.nSmallest(-1)]).toEqual([]); });

  test('providing n', () => { expect([...col.nSmallest(8)]).toEqual([0, 0, 1, 2, 3, 4, 5, 6]); });

  test('n longer than length of series', () => {
    expect([...col.nSmallest(25)]).toEqual([0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  });

  test('keep last duplicate option', () => {
    expect([...col.nSmallest(25, 'last')]).toEqual([0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);

    expect([...col.nSmallest(-5, 'last')]).toEqual([]);
  });

  test('keep none duplicate option throws',
       () => { expect(() => col.nSmallest(25, 'none')).toThrow(); });
});

test('Series.scatter (series)', () => {
  const col     = Series.new({type: new Int32, data: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]});
  const values  = Series.new({type: new Int32, data: [200, 400, 500, 800]});
  const indices = Series.new({type: new Int32, data: [2, 4, 5, 8]});

  const result = col.scatter(values, indices);

  expect([...result]).toEqual([0, 1, 200, 3, 400, 500, 6, 7, 800, 9]);
});

test('Series.scatter (series with array indices)', () => {
  const col     = Series.new({type: new Int32, data: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]});
  const values  = Series.new({type: new Int32, data: [200, 400, 500, 800]});
  const indices = [2, 4, 5, 8];

  const result = col.scatter(values, indices);

  expect([...result]).toEqual([0, 1, 200, 3, 400, 500, 6, 7, 800, 9]);
});

test('Series.scatter (scalar)', () => {
  const col     = Series.new({type: new Int32, data: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]});
  const indices = Series.new({type: new Int32, data: [2, 4, 5, 8]});

  const result = col.scatter(999, indices);

  expect([...result]).toEqual([0, 1, 999, 3, 999, 999, 6, 7, 999, 9]);
});

test('Series.scatter (scalar with array indicies)', () => {
  const col     = Series.new({type: new Int32, data: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]});
  const indices = [2, 4, 5, 8];

  const result = col.scatter(999, indices);

  expect([...result]).toEqual([0, 1, 999, 3, 999, 999, 6, 7, 999, 9]);
});

test('Series.scatter (check_bounds)', () => {
  const col          = Series.new({type: new Int32, data: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]});
  const values       = Series.new({type: new Int32, data: [200, 400, 500, 800]});
  const good_indices = [2, 4, 5, 8];
  const bad_indices  = [2, 4, 5, 18];

  const result = col.scatter(values, good_indices, true)
                   .scatter(999, good_indices, true)
                   .scatter(values, bad_indices)
                   .scatter(999, bad_indices);

  expect(() => result.scatter(values, bad_indices, true)).toThrowError();
  expect(() => result.scatter(999, bad_indices, true)).toThrowError();
});

test('Series.scatter (scalar)', () => {
  const col     = Series.new({type: new Int32, data: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]});
  const indices = Series.new({type: new Int32, data: [2, 4, 5, 8]});

  const result = col.scatter(999, indices);

  expect([...result]).toEqual([0, 1, 999, 3, 999, 999, 6, 7, 999, 9]);
});

test('Series.filter', () => {
  const col = Series.new([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);

  const mask = Series.new([false, false, true, false, true, true, false, false, true, false]);

  const result = col.filter(mask);

  const expected = Series.new({type: new Int32, data: new Int32Buffer([2, 4, 5, 8])});
  expect([...result]).toEqual([...expected]);
});

describe('toArrow()', () => {
  test('converts Uint8 Series to Uint8Vector', () => {
    const uint8Col = Series.new({type: new Uint8, data: new Uint8Buffer(Buffer.from('hello'))});
    const uint8Vec = uint8Col.toArrow();
    expect(uint8Vec).toBeInstanceOf(Uint8Vector);
    expect([...uint8Vec]).toEqual([...Buffer.from('hello')]);
  });
  test('converts String Series to Utf8Vector', () => {
    const utf8Col    = Series.new({type: new Uint8, data: new Uint8Buffer(Buffer.from('hello'))});
    const offsetsCol = Series.new({type: new Int32, data: new Int32Buffer([0, utf8Col.length])});
    const stringsCol = Series.new({
      type: new Utf8String(),
      length: 1,
      nullMask: new Uint8Buffer([255]),
      children: [offsetsCol, utf8Col],
    });
    const utf8Vec    = stringsCol.toArrow();
    expect(utf8Vec).toBeInstanceOf(Utf8Vector);
    expect([...utf8Vec]).toEqual(['hello']);
  });
});

test('Series.orderBy (ascending, non-null)', () => {
  const col    = Series.new({type: new Int32, data: new Int32Buffer([1, 3, 5, 4, 2, 0])});
  const result = col.orderBy(true, 'before');

  const expected = [5, 0, 4, 1, 3, 2];
  expect([...result]).toEqual(expected);
});

test('Series.orderBy (descending, non-null)', () => {
  const col    = Series.new({type: new Int32, data: new Int32Buffer([1, 3, 5, 4, 2, 0])});
  const result = col.orderBy(false, 'before');

  const expected = [2, 3, 1, 4, 0, 5];
  expect([...result]).toEqual(expected);
});

test('Series.orderBy (ascending, null before)', () => {
  const mask = new Uint8Buffer(BoolVector.from([1, 0, 1, 1, 1, 1]).values);
  const col =
    Series.new({type: new Int32, data: new Int32Buffer([1, 3, 5, 4, 2, 0]), nullMask: mask});
  const result = col.orderBy(true, 'before');

  const expected = [1, 5, 0, 4, 3, 2];
  expect([...result]).toEqual(expected);
});

test('Series.orderBy (ascending, null after)', () => {
  const mask = new Uint8Buffer(BoolVector.from([1, 0, 1, 1, 1, 1]).values);
  const col =
    Series.new({type: new Int32, data: new Int32Buffer([1, 3, 5, 4, 2, 0]), nullMask: mask});
  const result = col.orderBy(true, 'after');

  const expected = [5, 0, 4, 3, 2, 1];
  expect([...result]).toEqual(expected);
});

test('Series.orderBy (descendng, null before)', () => {
  const mask = new Uint8Buffer(BoolVector.from([1, 0, 1, 1, 1, 1]).values);
  const col =
    Series.new({type: new Int32, data: new Int32Buffer([1, 3, 5, 4, 2, 0]), nullMask: mask});
  const result = col.orderBy(false, 'before');

  const expected = [2, 3, 4, 0, 5, 1];

  expect([...result]).toEqual(expected);
});

test('Series.orderBy (descending, null after)', () => {
  const mask = new Uint8Buffer(BoolVector.from([1, 0, 1, 1, 1, 1]).values);
  const col =
    Series.new({type: new Int32, data: new Int32Buffer([1, 3, 5, 4, 2, 0]), nullMask: mask});
  const result = col.orderBy(false, 'after');

  const expected = [1, 2, 3, 4, 0, 5];
  expect([...result]).toEqual(expected);
});

test('Series.sortValues (ascending)', () => {
  const col    = Series.new({type: new Int32, data: new Int32Buffer([1, 3, 5, 4, 2, 0])});
  const result = col.sortValues();

  const expected = [0, 1, 2, 3, 4, 5];
  expect([...result]).toEqual(expected);
});

test('Series.sortValues (descending)', () => {
  const col    = Series.new({type: new Int32, data: new Int32Buffer([1, 3, 5, 4, 2, 0])});
  const result = col.sortValues(false);

  const expected = [5, 4, 3, 2, 1, 0];
  expect([...result]).toEqual(expected);
});

test('Series.isNull (numeric)', () => {
  const col    = Series.new({type: new Int32, data: [0, 1, null, 3, 4, null, 6, null]});
  const result = col.isNull();

  const expected = [false, false, true, false, false, true, false, true];
  expect([...result]).toEqual(expected);
});

test('Series.isNotNull (numeric)', () => {
  const col    = Series.new({type: new Int32, data: [0, 1, null, 3, 4, null, 6, null]});
  const result = col.isNotNull();

  const expected = [true, true, false, true, true, false, true, false];
  expect([...result]).toEqual(expected);
});

test('Series.dropNulls (drop nulls only)', () => {
  const mask = new Uint8Buffer(BoolVector.from([0, 1, 1, 1, 1, 0]).values);
  const col =
    Series.new({type: new Float32, data: new Float32Buffer([1, 3, NaN, 4, 2, 0]), nullMask: mask});
  const result = col.dropNulls();

  const expected = [3, NaN, 4, 2];
  expect([...result]).toEqual(expected);
});

test('FloatSeries.dropNaNs (drop NaN values only)', () => {
  const mask = new Uint8Buffer(BoolVector.from([0, 1, 1, 1, 1, 0]).values);
  const col =
    Series.new({type: new Float32, data: new Float32Buffer([1, 3, NaN, 4, 2, 0]), nullMask: mask});
  const result = col.dropNaNs();

  const expected = [null, 3, 4, 2, null];
  expect([...result]).toEqual(expected);
});

test('Series.countNonNulls', () => {
  const twoNonNulls  = Series.new(['foo', null, 'bar']);
  const fourNonNulls = Series.new([NaN, null, 10, 15, 17, null]);
  const fiveNonNulls = Series.new([0, 1, null, 3, 4, null, 6, null]);

  expect(twoNonNulls.countNonNulls()).toEqual(2);
  expect(fourNonNulls.countNonNulls()).toEqual(4);
  expect(fiveNonNulls.countNonNulls()).toEqual(5);
});

test('FloatSeries.nansToNulls', () => {
  const col = Series.new({type: new Float32, data: new Float32Buffer([1, 3, NaN, 4, 2, 0])});

  const result = col.nansToNulls();

  const expected = [1, 3, null, 4, 2, 0];
  expect([...result]).toEqual(expected);
  expect(result.nullCount).toEqual(1);
  expect(col.nullCount).toEqual(0);
});

test('Series.reverse', () => {
  const array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
  const col   = Series.new(array);

  expect([...col.reverse()]).toEqual(array.reverse());
});

describe.each([new Int32, new Float32, new Float64])('Series.sequence({type=%p,, ...})', (typ) => {
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

test('Series.valueCounts', () => {
  const s      = Series.new({type: new Int32, data: [110, 120, 100, 110, 120, 120]});
  const result = s.valueCounts();
  const count  = [...result.count];
  const value  = [...result.value];

  const countMap: Record<number, number> = {100: 1, 110: 2, 120: 3};

  for (let i = 0; i < value.length; i++) {
    const currentVal   = value[i] as number;
    const currentCount = count[i];
    expect(currentCount).toBe(countMap[currentVal]);
  }
});

test.each`
nullsEqual        | data                           | expected
${true}         | ${[null, null, 1, 2, 3, 4, 4]} | ${[null, 1, 2, 3, 4]}
${false}       | ${[null, null, 1, 2, 3, 4, 4]} | ${[null, null, 1, 2, 3, 4]}
`('Series.unique($nullsEqual)', ({nullsEqual, data, expected}) => {
  const s      = Series.new({type: new Int32, data});
  const result = s.unique(nullsEqual);
  expect([...result]).toEqual(expected);
});

test.each`
data | replaceValue | expected
${[1, null, 3]} | ${Series.new([9, 9, 9])} | ${[1, 9, 3]}
${['foo', 'bar', null]} | ${Series.new(['test','test','test'])} | ${['foo', 'bar', 'test']}
${[true, false, null]} | ${Series.new([false, false, false])} | ${[true, false, false]}
${[1, null, 3]} | ${9} | ${[1, 9, 3]}
${['foo', 'bar', null]} | ${'test'} | ${['foo', 'bar', 'test']}
${[true, false, null]} | ${false} | ${[true, false, false]}
`('Series.replaceNulls', ({data, replaceValue, expected}) => {
  const s       = Series.new(data);
  const result  = s.replaceNulls(replaceValue);

  expect([...result]).toEqual(expected);
});

test.each`
data | expected
${[1, null, 3]} | ${[1, 1, 3]}
${['foo', 'bar', null]} | ${['foo', 'bar', 'bar']}
${[true, false, null]} | ${[true, false, false]}
`('Series.replaceNullsPreceding', ({data, expected})=> {
  const s       = Series.new(data);
  const result  = s.replaceNullsPreceding();

  expect([...result]).toEqual(expected);
});

test.each`
data | expected
${[1, null, 3]} | ${[1, 3, 3]}
${['foo', 'bar', null]} | ${['foo', 'bar', null]}
${[true, null, true]} | ${[true, true, true]}
`('Series.replaceNullsFollowing', ({data, expected})=> {
  const s       = Series.new(data);
  const result  = s.replaceNullsFollowing();

  expect([...result]).toEqual(expected);
});

test('Series.TimestampDay (Int32Buffer)', () => {
  const dateTime = Math.floor(new Date('May 13, 2021 16:38:30:100 GMT+00:00').getTime() / 86400000);
  const s        = Series.new({type: new TimestampDay, data: new Int32Buffer([dateTime])});
  const val      = s.getValue(0);

  expect(val?.getUTCFullYear()).toBe(2021);
  expect(val?.getUTCMonth()).toBe(4);
  expect(val?.getUTCDate()).toBe(13);
});

test('Series.TimestampSecond (Int64Buffer)', () => {
  const dateTime = Math.floor(new Date('May 13, 2021 16:38:30:100 GMT+00:00').getTime() / 1000);
  const s = Series.new({type: new TimestampSecond, data: new Int64Buffer([dateTime].map(BigInt))});
  const val = s.getValue(0);

  expect(val?.getUTCFullYear()).toBe(2021);
  expect(val?.getUTCMonth()).toBe(4);
  expect(val?.getUTCDate()).toBe(13);
  expect(val?.getUTCHours()).toBe(16);
  expect(val?.getUTCMinutes()).toBe(38);
  expect(val?.getUTCSeconds()).toBe(30);
});

test('Series.TimestampMillisecond (Int64Buffer)', () => {
  const dateTime = new Date('May 13, 2021 16:38:30:100 GMT+00:00').getTime();
  const s =
    Series.new({type: new TimestampMillisecond, data: new Int64Buffer([dateTime].map(BigInt))});
  const val = s.getValue(0);

  expect(val?.getUTCFullYear()).toBe(2021);
  expect(val?.getUTCMonth()).toBe(4);
  expect(val?.getUTCDate()).toBe(13);
  expect(val?.getUTCHours()).toBe(16);
  expect(val?.getUTCMinutes()).toBe(38);
  expect(val?.getUTCSeconds()).toBe(30);
  expect(val?.getUTCMilliseconds()).toBe(100);
});

test('Series.TimestampNanosecond (Int64Buffer)', () => {
  const dateTime = new Date('May 13, 2021 16:38:30:100 GMT+00:00').getTime() * 1000;
  const s =
    Series.new({type: new TimestampMicrosecond, data: new Int64Buffer([dateTime].map(BigInt))});
  const val = s.getValue(0);

  expect(val?.getUTCFullYear()).toBe(2021);
  expect(val?.getUTCMonth()).toBe(4);
  expect(val?.getUTCDate()).toBe(13);
  expect(val?.getUTCHours()).toBe(16);
  expect(val?.getUTCMinutes()).toBe(38);
  expect(val?.getUTCSeconds()).toBe(30);
  expect(val?.getUTCMilliseconds()).toBe(100);
});

test('Series.TimestampMicrosecond (Int64Buffer)', () => {
  const dateTime = new Date('May 13, 2021 16:38:30:100 GMT+00:00').getTime() * 1000000;
  const s =
    Series.new({type: new TimestampNanosecond, data: new Int64Buffer([dateTime].map(BigInt))});
  const val = s.getValue(0);

  expect(val?.getUTCFullYear()).toBe(2021);
  expect(val?.getUTCMonth()).toBe(4);
  expect(val?.getUTCDate()).toBe(13);
  expect(val?.getUTCHours()).toBe(16);
  expect(val?.getUTCMinutes()).toBe(38);
  expect(val?.getUTCSeconds()).toBe(30);
  expect(val?.getUTCMilliseconds()).toBe(100);
});

test('Series.TimestampDay', () => {
  const dateTime = new Date('May 13, 2021 16:38:30:100 GMT+00:00');
  const s        = Series.new({type: new TimestampDay, data: [dateTime]});
  const val      = s.getValue(0);

  expect(val?.getUTCFullYear()).toBe(2021);
  expect(val?.getUTCMonth()).toBe(4);
  expect(val?.getUTCDate()).toBe(13);
});

test('Series.TimestampSecond', () => {
  const dateTime = new Date('May 13, 2021 16:38:30:100 GMT+00:00');
  const s        = Series.new({type: new TimestampSecond, data: [dateTime]});
  const val      = s.getValue(0);

  expect(val?.getUTCFullYear()).toBe(2021);
  expect(val?.getUTCMonth()).toBe(4);
  expect(val?.getUTCDate()).toBe(13);
  expect(val?.getUTCHours()).toBe(16);
  expect(val?.getUTCMinutes()).toBe(38);
  expect(val?.getUTCSeconds()).toBe(30);
});

test('Series.TimestampMillisecond', () => {
  const dateTime = new Date('May 13, 2021 16:38:30:100 GMT+00:00');
  const s        = Series.new({type: new TimestampMillisecond, data: [dateTime]});
  const val      = s.getValue(0);

  expect(val?.getUTCFullYear()).toBe(2021);
  expect(val?.getUTCMonth()).toBe(4);
  expect(val?.getUTCDate()).toBe(13);
  expect(val?.getUTCHours()).toBe(16);
  expect(val?.getUTCMinutes()).toBe(38);
  expect(val?.getUTCSeconds()).toBe(30);
  expect(val?.getUTCMilliseconds()).toBe(100);
});

test('Series.TimestampNanosecond', () => {
  const dateTime = new Date('May 13, 2021 16:38:30:100 GMT+00:00');
  const s        = Series.new({type: new TimestampMicrosecond, data: [dateTime]});
  const val      = s.getValue(0);

  expect(val?.getUTCFullYear()).toBe(2021);
  expect(val?.getUTCMonth()).toBe(4);
  expect(val?.getUTCDate()).toBe(13);
  expect(val?.getUTCHours()).toBe(16);
  expect(val?.getUTCMinutes()).toBe(38);
  expect(val?.getUTCSeconds()).toBe(30);
  expect(val?.getUTCMilliseconds()).toBe(100);
});

test('Series.TimestampMicrosecond', () => {
  const dateTime = new Date('May 13, 2021 16:38:30:100 GMT+00:00');
  const s        = Series.new({type: new TimestampNanosecond, data: [dateTime]});
  const val      = s.getValue(0);

  expect(val?.getUTCFullYear()).toBe(2021);
  expect(val?.getUTCMonth()).toBe(4);
  expect(val?.getUTCDate()).toBe(13);
  expect(val?.getUTCHours()).toBe(16);
  expect(val?.getUTCMinutes()).toBe(38);
  expect(val?.getUTCSeconds()).toBe(30);
  expect(val?.getUTCMilliseconds()).toBe(100);
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

test.each`
keep     | nullsEqual | nullsFirst | data                           | expected
${true}  | ${true}    | ${true}    | ${[4, null, 1, 2, null, 3, 4]} | ${[null, 1, 2, 3, 4]}
${false} | ${true}    | ${true}    | ${[4, null, 1, 2, null, 3, 4]} | ${[1, 2, 3]}
${true}  | ${true}    | ${false}   | ${[4, null, 1, 2, null, 3, 4]} | ${[1, 2, 3, 4, null]}
${false} | ${true}    | ${false}   | ${[4, null, 1, 2, null, 3, 4]} | ${[1, 2, 3]}
${true}  | ${false}   | ${true}    | ${[4, null, 1, 2, null, 3, 4]} | ${[null, null, 1, 2, 3, 4]}
${false} | ${false}   | ${true}    | ${[4, null, 1, 2, null, 3, 4]} | ${[null, null, 1, 2, 3]}
${true}  | ${false}   | ${false}   | ${[4, null, 1, 2, null, 3, 4]} | ${[1, 2, 3, 4, null, null]}
${false} | ${false}   | ${false}   | ${[4, null, 1, 2, null, 3, 4]} | ${[1, 2, 3, null, null]}
`('Series.dropDuplicates($keep, $nullsEqual, $nullsFirst)', ({keep, nullsEqual, nullsFirst, data, expected}) => {
  const s      = Series.new({type: new Int32, data});
  const result = s.dropDuplicates(keep, nullsEqual, nullsFirst);
  expect([...result]).toEqual(expected);
});

describe('Series.readText', () => {
  test('can read a json file', async () => {
    const rows = [
      {a: 0, b: 1.0, c: '2'},
      {a: 1, b: 2.0, c: '3'},
      {a: 2, b: 3.0, c: '4'},
    ];
    const outputString = JSON.stringify(rows);
    const path         = Path.join(readTextTmpDir, 'simple.txt');
    await promises.writeFile(path, outputString);
    const text = Series.readText(path, '');
    expect(text.getValue(0)).toEqual(outputString);
    await new Promise<void>((resolve, reject) =>
                              rimraf(path, (err?: Error|null) => err ? reject(err) : resolve()));
  });
  test('can read a random file', async () => {
    const outputString = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()';
    const path         = Path.join(readTextTmpDir, 'simple.txt');
    await promises.writeFile(path, outputString);
    const text = Series.readText(path, '');
    expect(text.getValue(0)).toEqual(outputString);
    await new Promise<void>((resolve, reject) =>
                              rimraf(path, (err?: Error|null) => err ? reject(err) : resolve()));
  });
  test('can read an empty file', async () => {
    const outputString = '';
    const path         = Path.join(readTextTmpDir, 'simple.txt');
    await promises.writeFile(path, outputString);
    const text = Series.readText(path, '');
    expect(text.getValue(0)).toEqual(outputString);
    await new Promise<void>((resolve, reject) =>
                              rimraf(path, (err?: Error|null) => err ? reject(err) : resolve()));
  });
});

describe('StringSeries split', () => {
  test('split a basic string', () => {
    const input   = StringSeries.new(['abcdefg']);
    const example = StringSeries.new(['abcd', 'efg']);
    const result  = StringSeries.new(input._col.split('d'));
    expect(result).toEqual(example);
  });
  test('split a string twice', () => {
    const input   = StringSeries.new(['abcdefgdcba']);
    const example = StringSeries.new(['abcd', 'efgd', 'cba']);
    const result  = StringSeries.new(input._col.split('d'));
    expect(result).toEqual(example);
  });
});

let readTextTmpDir = '';

const rimraf = require('rimraf');

beforeAll(async () => {  //
  readTextTmpDir = await promises.mkdtemp(Path.join('/tmp', 'node_cudf'));
});

afterAll(() => {
  return new Promise<void>((resolve, reject) => {  //
    rimraf(readTextTmpDir, (err?: Error|null) => err ? reject(err) : resolve());
  });
});
