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
  Float32Buffer,
  Int32Buffer,
  Int64Buffer,
  setDefaultAllocator,
  Uint8Buffer
} from '@nvidia/cuda';
import {
  Bool8,
  Column,
  Float32,
  Float64,
  Int32,
  Int64,
  Series,
  TimestampDay,
  TimestampMicrosecond,
  TimestampMillisecond,
  TimestampNanosecond,
  TimestampSecond,
  Uint8,
  Utf8String
} from '@rapidsai/cudf';
import {CudaMemoryResource, DeviceBuffer} from '@rapidsai/rmm';
import {Uint8Vector, Utf8Vector} from 'apache-arrow';
import {BoolVector} from 'apache-arrow';

const mr = new CudaMemoryResource();

setDefaultAllocator((byteLength: number) => new DeviceBuffer(byteLength, mr));

test('Series initialization with properties (non-null)', () => {
  const length = 100;
  const s      = Series.new({type: new Int32, data: new Int32Buffer(length)});

  expect(s.type).toBeInstanceOf(Int32);
  expect(s.length).toBe(length);
  expect(s.nullCount).toBe(0);
  expect(s.hasNulls).toBe(false);
  expect(s.nullable).toBe(false);
});

test('Series initialization with properties (null)', () => {
  const length = 10;
  const s      = Series.new({
    type: new Int32,
    data: new Int32Buffer(length),
    nullMask: new Uint8Buffer([250, 255]),
  });

  expect(s.type).toBeInstanceOf(Int32);
  expect(s.length).toBe(length);
  expect(s.nullCount).toBe(2);
  expect(s.hasNulls).toBe(true);
  expect(s.nullable).toBe(true);
});

test('Series initialization with Column', () => {
  const length = 100;
  const col    = new Column({type: new Int32, data: new Int32Buffer(length)});
  const s      = Series.new(col);

  expect(s.type).toBeInstanceOf(Int32);
  expect(s.length).toBe(length);
  expect(s.nullCount).toBe(0);
  expect(s.hasNulls).toBe(false);
  expect(s.nullable).toBe(false);
});

test('Series initialization with Array of mixed values', () => {
  const s = Series.new({type: new Int32, data: [0, 1, null, 2]});

  expect(s.type).toBeInstanceOf(Int32);
  expect(s.length).toBe(4);
  expect(s.nullCount).toBe(1);
  expect(s.hasNulls).toBe(true);
  expect(s.nullable).toBe(true);
  expect([...s]).toEqual([0, 1, null, 2]);
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

test('Series.gather', () => {
  const col = Series.new({type: new Int32, data: new Int32Buffer([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])});

  const selection = Series.new({type: new Int32, data: new Int32Buffer([2, 4, 5, 8])});

  const result = col.gather(selection);

  expect([...result]).toEqual([...selection]);
});

describe('Series.head', () => {
  test('default index', () => {
    const col =
      Series.new({type: new Int32, data: new Int32Buffer([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])});
    expect([...col.head()]).toEqual([0, 1, 2, 3, 4]);
  });

  test('invalid index', () => {
    const col =
      Series.new({type: new Int32, data: new Int32Buffer([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])});
    expect(() => col.head(-1)).toThrowError();
  });

  test('providing index', () => {
    const col =
      Series.new({type: new Int32, data: new Int32Buffer([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])});
    expect([...col.head(8)]).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
  });

  test('index longer than length of series', () => {
    const col =
      Series.new({type: new Int32, data: new Int32Buffer([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])});
    expect([...col.head(25)]).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  });
});

describe('Series.tail', () => {
  const col = Series.new({type: new Int32, data: new Int32Buffer([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])});

  test('default index', () => { expect([...col.tail()]).toEqual([5, 6, 7, 8, 9]); });

  test('invalid index', () => { expect(() => col.tail(-1)).toThrowError(); });

  test('providing index', () => { expect([...col.tail(8)]).toEqual([2, 3, 4, 5, 6, 7, 8, 9]); });

  test('index longer than length of series', () => {
    expect([...col.tail(25)]).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  });
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
  const result = col.orderBy(true, 'BEFORE');

  const expected = [5, 0, 4, 1, 3, 2];
  expect([...result]).toEqual(expected);
});

test('Series.orderBy (descending, non-null)', () => {
  const col    = Series.new({type: new Int32, data: new Int32Buffer([1, 3, 5, 4, 2, 0])});
  const result = col.orderBy(false, 'BEFORE');

  const expected = [2, 3, 1, 4, 0, 5];
  expect([...result]).toEqual(expected);
});

test('Series.orderBy (ascending, null before)', () => {
  const mask = new Uint8Buffer(BoolVector.from([1, 0, 1, 1, 1, 1]).values);
  const col =
    Series.new({type: new Int32, data: new Int32Buffer([1, 3, 5, 4, 2, 0]), nullMask: mask});
  const result = col.orderBy(true, 'BEFORE');

  const expected = [1, 5, 0, 4, 3, 2];
  expect([...result]).toEqual(expected);
});

test('Series.orderBy (ascending, null after)', () => {
  const mask = new Uint8Buffer(BoolVector.from([1, 0, 1, 1, 1, 1]).values);
  const col =
    Series.new({type: new Int32, data: new Int32Buffer([1, 3, 5, 4, 2, 0]), nullMask: mask});
  const result = col.orderBy(true, 'AFTER');

  const expected = [5, 0, 4, 3, 2, 1];
  expect([...result]).toEqual(expected);
});

test('Series.orderBy (descendng, null before)', () => {
  const mask = new Uint8Buffer(BoolVector.from([1, 0, 1, 1, 1, 1]).values);
  const col =
    Series.new({type: new Int32, data: new Int32Buffer([1, 3, 5, 4, 2, 0]), nullMask: mask});
  const result = col.orderBy(false, 'BEFORE');

  const expected = [2, 3, 4, 0, 5, 1];

  expect([...result]).toEqual(expected);
});

test('Series.orderBy (descending, null after)', () => {
  const mask = new Uint8Buffer(BoolVector.from([1, 0, 1, 1, 1, 1]).values);
  const col =
    Series.new({type: new Int32, data: new Int32Buffer([1, 3, 5, 4, 2, 0]), nullMask: mask});
  const result = col.orderBy(false, 'AFTER');

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

test('Series.value_counts', () => {
  const s      = Series.new({type: new Int32, data: [110, 120, 100, 110, 120, 120]});
  const result = s.value_counts();
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
nulls_equal        | data                           | expected
${true}         | ${[null, null, 1, 2, 3, 4, 4]} | ${[null, 1, 2, 3, 4]}
${false}       | ${[null, null, 1, 2, 3, 4, 4]} | ${[null, null, 1, 2, 3, 4]}
`('Series.unique($nulls_equal)', ({nulls_equal, data, expected}) => {
  const s      = Series.new({type: new Int32, data});
  const result = s.unique(nulls_equal);
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
