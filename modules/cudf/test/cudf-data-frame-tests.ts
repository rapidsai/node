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

import {Float32Buffer, Int32Buffer, setDefaultAllocator, Uint8Buffer} from '@nvidia/cuda';
import {
  Bool8,
  DataFrame,
  Float32,
  GroupByMultiple,
  GroupBySingle,
  Int32,
  Int8,
  Series,
  Table
} from '@rapidsai/cudf';
import {CudaMemoryResource, DeviceBuffer} from '@rapidsai/rmm';
import {BoolVector} from 'apache-arrow';

const mr = new CudaMemoryResource();

setDefaultAllocator((byteLength: number) => new DeviceBuffer(byteLength, mr));

test('DataFrame initialization', () => {
  const length = 100;
  const col_0  = Series.new({type: new Int32(), data: new Int32Buffer(length)});

  const col_1   = Series.new({
    type: new Bool8(),
    data: new Uint8Buffer(length),
    nullMask: new Uint8Buffer(64),
  });
  const table_0 = new DataFrame({'col_0': col_0, 'col_1': col_1});
  expect(table_0.numColumns).toBe(2);
  expect(table_0.numRows).toBe(length);
  expect(table_0.names).toStrictEqual(['col_0', 'col_1']);
  expect(table_0.get('col_0').type.typeId).toBe(col_0.type.typeId);
  expect(table_0.get('col_1').type.typeId).toBe(col_1.type.typeId);
});

test('DataFrame asTable', () => {
  const length = 100;
  const col_0  = Series.new({type: new Int32(), data: new Int32Buffer(length)});

  const col_1   = Series.new({
    type: new Bool8(),
    data: new Uint8Buffer(length),
    nullMask: new Uint8Buffer(64),
  });
  const table_0 = new DataFrame({'col_0': col_0, 'col_1': col_1});
  expect(table_0.asTable()).toBeInstanceOf(Table);
});

test('DataFrame.get', () => {
  const length = 100;
  const col_0  = Series.new({type: new Int32(), data: new Int32Buffer(length)});

  const col_1   = Series.new({
    type: new Bool8(),
    data: new Uint8Buffer(length),
    nullMask: new Uint8Buffer(64),
  });
  const table_0 = new DataFrame({'col_0': col_0, 'col_1': col_1});
  expect(table_0.get('col_0').type.typeId).toBe(col_0.type.typeId);
  expect(() => { (<any>table_0).get(2); }).toThrow();
  expect(() => { (<any>table_0).get('junk'); }).toThrow();
});

test('DataFrame.select', () => {
  const length = 100;
  const col_0  = Series.new({type: new Int32(), data: new Int32Buffer(length)});

  const col_1 = Series.new({
    type: new Bool8(),
    data: new Uint8Buffer(length),
    nullMask: new Uint8Buffer(64),
  });

  const col_2 = Series.new({type: new Int32(), data: new Int32Buffer(length)});
  const col_3 = Series.new({type: new Int32(), data: new Int32Buffer(length)});

  const table_0 = new DataFrame({'col_0': col_0, 'col_1': col_1, 'col_2': col_2, 'col_3': col_3});

  expect(table_0.numColumns).toBe(4);
  expect(table_0.numRows).toBe(length);
  expect(table_0.names).toStrictEqual(['col_0', 'col_1', 'col_2', 'col_3']);

  expect(table_0.select(['col_0'])).toStrictEqual(new DataFrame({'col_0': col_0}));
  expect(table_0.select(['col_0', 'col_3']))
    .toStrictEqual(new DataFrame({'col_0': col_0, 'col_3': col_3}));
});

test('DataFrame.assign', () => {
  const length = 100;
  const col_0  = Series.new({type: new Int32(), data: new Int32Buffer(length)});

  const col_1 = Series.new({
    type: new Bool8(),
    data: new Uint8Buffer(length),
    nullMask: new Uint8Buffer(64),
  });

  const col_2 = Series.new({type: new Int32(), data: new Int32Buffer(length)});
  const col_3 = Series.new({type: new Int32(), data: new Int32Buffer(length)});
  const col_4 = Series.new({type: new Int32(), data: new Int32Buffer(length)});

  const table_0 = new DataFrame({'col_0': col_0, 'col_1': col_1, 'col_2': col_2});

  const table_1 = table_0.assign({'col_3': col_3});
  expect(table_1.numColumns).toBe(4);
  expect(table_1.numRows).toBe(length);
  expect(table_1.names).toStrictEqual(['col_0', 'col_1', 'col_2', 'col_3']);

  // testing DataFrame.assign(DataFrame)
  const table_2 = new DataFrame({'col_4': col_4});
  const table_3 = table_0.assign(table_2);
  expect(table_3.numColumns).toBe(4);
  expect(table_3.numRows).toBe(length);
  expect(table_3.names).toStrictEqual(['col_0', 'col_1', 'col_2', 'col_4']);
});

test('DataFrame.drop', () => {
  const length = 100;
  const col_0  = Series.new({type: new Int32(), data: new Int32Buffer(length)});

  const col_1 = Series.new({
    type: new Bool8(),
    data: new Uint8Buffer(length),
    nullMask: new Uint8Buffer(64),
  });

  const col_2 = Series.new({type: new Int32(), data: new Int32Buffer(length)});

  const table_0 = new DataFrame({'col_0': col_0, 'col_1': col_1, 'col_2': col_2});

  const table_1 = table_0.drop(['col_1']);
  expect(table_1.numColumns).toBe(2);
  expect(table_1.numRows).toBe(length);
  expect(table_1.names).toStrictEqual(['col_0', 'col_2']);
});

test('DataFrame.orderBy (ascending, non-null)', () => {
  const col    = Series.new({type: new Int32(), data: new Int32Buffer([1, 3, 5, 4, 2, 0])});
  const df     = new DataFrame({'a': col});
  const result = df.orderBy({'a': {ascending: true, null_order: 'before'}});

  const expected = [5, 0, 4, 1, 3, 2];
  expect([...result]).toEqual([...Buffer.from(expected)]);
});

test('DataFrame.orderBy (descending, non-null)', () => {
  const col    = Series.new({type: new Int32(), data: new Int32Buffer([1, 3, 5, 4, 2, 0])});
  const df     = new DataFrame({'a': col});
  const result = df.orderBy({'a': {ascending: false, null_order: 'before'}});

  const expected = [2, 3, 1, 4, 0, 5];
  expect([...result]).toEqual([...Buffer.from(expected)]);
});

test('DataFrame.orderBy (ascending, null before)', () => {
  const mask = new Uint8Buffer(BoolVector.from([1, 0, 1, 1, 1, 1]).values);
  const col =
    Series.new({type: new Int32(), data: new Int32Buffer([1, 3, 5, 4, 2, 0]), nullMask: mask});
  const df     = new DataFrame({'a': col});
  const result = df.orderBy({'a': {ascending: true, null_order: 'before'}});

  const expected = [1, 5, 0, 4, 3, 2];
  expect([...result]).toEqual([...Buffer.from(expected)]);
});

test('DataFrame.orderBy (ascending, null after)', () => {
  const mask = new Uint8Buffer(BoolVector.from([1, 0, 1, 1, 1, 1]).values);
  const col =
    Series.new({type: new Int32(), data: new Int32Buffer([1, 3, 5, 4, 2, 0]), nullMask: mask});
  const df     = new DataFrame({'a': col});
  const result = df.orderBy({'a': {ascending: true, null_order: 'after'}});

  const expected = [5, 0, 4, 3, 2, 1];
  expect([...result]).toEqual([...Buffer.from(expected)]);
});

test('DataFrame.orderBy (descendng, null before)', () => {
  const mask = new Uint8Buffer(BoolVector.from([1, 0, 1, 1, 1, 1]).values);
  const col =
    Series.new({type: new Int32(), data: new Int32Buffer([1, 3, 5, 4, 2, 0]), nullMask: mask});
  const df     = new DataFrame({'a': col});
  const result = df.orderBy({'a': {ascending: false, null_order: 'before'}});

  const expected = [2, 3, 4, 0, 5, 1];

  expect([...result]).toEqual([...Buffer.from(expected)]);
});

test('DataFrame.orderBy (descending, null after)', () => {
  const mask = new Uint8Buffer(BoolVector.from([1, 0, 1, 1, 1, 1]).values);
  const col =
    Series.new({type: new Int32(), data: new Int32Buffer([1, 3, 5, 4, 2, 0]), nullMask: mask});
  const df     = new DataFrame({'a': col});
  const result = df.orderBy({'a': {ascending: false, null_order: 'after'}});

  const expected = [1, 2, 3, 4, 0, 5];
  expect([...result]).toEqual([...Buffer.from(expected)]);
});

test('DataFrame.gather (indices)', () => {
  const a = Series.new({type: new Int32(), data: new Int32Buffer([0, 1, 2, 3, 4, 5])});
  const b =
    Series.new({type: new Float32(), data: new Float32Buffer([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])});
  const df = new DataFrame({'a': a, 'b': b});

  const selection = Series.new({type: new Int32(), data: new Int32Buffer([2, 4, 5])});

  const result = df.gather(selection);
  expect(result.numRows).toBe(3);

  const ra = result.get('a');
  const rb = result.get('b');

  const expected_a = Series.new({type: new Int32(), data: new Int32Buffer([2, 4, 5])});
  expect([...ra]).toEqual([...expected_a]);

  const expected_b = Series.new({type: new Float32(), data: new Float32Buffer([2.0, 4.0, 5.0])});
  expect([...rb]).toEqual([...expected_b]);
});

describe('Dataframe.head', () => {
  const a  = Series.new([0, 1, 2, 3, 4, 5, 6]);
  const b  = Series.new([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]);
  const c  = Series.new(['foo', null, null, 'bar', null, null, 'foo']);
  const df = new DataFrame({'a': a, 'b': b, 'c': c});

  test('default n', () => {
    const result = df.head();
    expect(result.numRows).toEqual(5);
    expect([...result.get('a')]).toEqual([0, 1, 2, 3, 4]);
    expect([...result.get('b')]).toEqual([0.0, 1.1, 2.2, 3.3, 4.4]);
    expect([...result.get('c')]).toEqual(['foo', null, null, 'bar', null]);
  });

  test('invalid n', () => { expect(() => df.head(-1)).toThrowError(); });

  test('providing n', () => {
    const result = df.head(2);
    expect(result.numRows).toEqual(2);
    expect([...result.get('a')]).toEqual([0, 1]);
    expect([...result.get('b')]).toEqual([0.0, 1.1]);
    expect([...result.get('c')]).toEqual(['foo', null]);
  });

  test('n longer than length of series', () => {
    const result = df.head(25);
    expect(result.numRows).toEqual(7);
    expect([...result.get('a')]).toEqual([...a]);
    expect([...result.get('b')]).toEqual([...b]);
    expect([...result.get('c')]).toEqual([...c]);
  });
});

describe('Dataframe.tail', () => {
  const a  = Series.new([0, 1, 2, 3, 4, 5, 6]);
  const b  = Series.new([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]);
  const c  = Series.new(['foo', null, null, 'bar', null, null, 'foo']);
  const df = new DataFrame({'a': a, 'b': b, 'c': c});

  test('default n', () => {
    const result = df.tail();
    expect(result.numRows).toEqual(5);
    expect([...result.get('a')]).toEqual([2, 3, 4, 5, 6]);
    expect([...result.get('b')]).toEqual([2.2, 3.3, 4.4, 5.5, 6.6]);
    expect([...result.get('c')]).toEqual([null, 'bar', null, null, 'foo']);
  });

  test('invalid n', () => { expect(() => df.tail(-1)).toThrowError(); });

  test('providing n', () => {
    const result = df.tail(2);
    expect(result.numRows).toEqual(2);
    expect([...result.get('a')]).toEqual([5, 6]);
    expect([...result.get('b')]).toEqual([5.5, 6.6]);
    expect([...result.get('c')]).toEqual([null, 'foo']);
  });

  test('n longer than length of series', () => {
    const result = df.tail(25);
    expect(result.numRows).toEqual(7);
    expect([...result.get('a')]).toEqual([...a]);
    expect([...result.get('b')]).toEqual([...b]);
    expect([...result.get('c')]).toEqual([...c]);
  });
});

test('DataFrame groupBy (single)', () => {
  const a   = Series.new({type: new Int32, data: [1, 2, 3, 1, 2, 2, 1, 3, 3, 2]});
  const b   = Series.new({type: new Float32, data: [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]});
  const df  = new DataFrame({'a': a, 'b': b});
  const out = df.groupBy({by: 'a'});
  expect(out instanceof GroupBySingle).toBe(true);
});

test('DataFrame groupBy (single)', () => {
  const a   = Series.new({type: new Int32, data: [1, 2, 3, 1, 2, 2, 1, 3, 3, 2]});
  const aa  = Series.new({type: new Int32, data: [1, 2, 3, 1, 2, 2, 1, 3, 3, 2]});
  const b   = Series.new({type: new Float32, data: [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]});
  const df  = new DataFrame({'a': a, 'aa': aa, 'b': b});
  const out = df.groupBy({by: ['a', 'aa'], index_key: 'ind'});
  expect(out instanceof GroupByMultiple).toBe(true);
});

test('DataFrame filter', () => {
  const a = Series.new({type: new Int32(), data: new Int32Buffer([0, 1, 2, 3, 4, 5])});
  const b =
    Series.new({type: new Float32(), data: new Float32Buffer([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])});
  const df = new DataFrame({'a': a, 'b': b});

  const mask =
    Series.new({length: 6, type: new Bool8(), data: new Uint8Buffer([0, 0, 1, 0, 1, 1])});

  const result = df.filter(mask);
  expect(result.numRows).toBe(3);

  const ra = result.get('a');
  const rb = result.get('b');

  const expected_a = Series.new({type: new Int32(), data: new Int32Buffer([2, 4, 5])});
  expect([...ra]).toEqual([...expected_a]);

  const expected_b = Series.new({type: new Float32(), data: new Float32Buffer([2.0, 4.0, 5.0])});
  expect([...rb]).toEqual([...expected_b]);
});

test(
  'dataframe.dropNulls(axis=0, thresh=df.numColumns), drop rows with non-null values < numColumn (drop row if atleast one null)',
  () => {
    const a  = Series.new({
      type: new Float32,
      data: [null, 1, null, null, null, null],
    });
    const b  = Series.new({
      type: new Float32,
      data: [null, 1, null, null, null, null],
    });
    const c  = Series.new({
      type: new Float32,
      data: [1, null, 3, 4, 5, 6],
    });
    const df = new DataFrame({'a': a, 'b': b, 'c': c});

    // all rows are dropped, since every row contains atleast one Null value
    const result = df.dropNulls(0, df.numColumns);
    expect(result.numRows).toEqual(0);
  });

test(
  'dataframe.dropNulls(axis=0, thresh=1), drop rows with non-null values < 1 (drop row if all null)',
  () => {
    const a  = Series.new({
      type: new Float32,
      data: [null, 1, null, null, null, null],
    });
    const b  = Series.new({
      type: new Float32,
      data: [null, 1, null, null, null, null],
    });
    const c  = Series.new({
      type: new Float32,
      data: [null, 2, 3, 4, 5, 6],
    });
    const df = new DataFrame({'a': a, 'b': b, 'c': c});

    const expected_a = Series.new({type: new Float32, data: [1, null, null, null, null]});
    const expected_c = Series.new({type: new Float32, data: [2, 3, 4, 5, 6]});

    // row 1 is dropped as it contains all Nulls
    const result = df.dropNulls(0, 1);
    const ra     = result.get('a');
    const rc     = result.get('c');

    expect([...ra]).toEqual([...expected_a]);
    expect([...rc]).toEqual([...expected_c]);
    expect(result.numRows).toEqual(5);
  });

test(
  'dataframe.dropNulls(axis=1, thresh=1), drop columns with non-null values < 1 (drop if all null)',
  () => {
    const a  = Series.new({
      type: new Float32,
      data: [null, 1, null, null, null, null],
    });
    const b  = Series.new({
      type: new Float32,
      data: [null, 1, 2, 3, 4, null],
    });
    const c  = Series.new({
      type: new Float32,
      data: [null, null, null, null, null, null],
    });
    const df = new DataFrame({'a': a, 'b': b, 'c': c});

    const result = df.dropNulls(1, 1);

    // column c is dropped as it contains all Null values
    expect(result.numColumns).toEqual(2);
    expect(result.names).toEqual(['a', 'b']);
  });

test(
  'dataframe.dropNulls(axis=1, thresh=df.numRows), drop columns with non-ull values < numRows (drop if atleast one null)',
  () => {
    const a  = Series.new({type: new Float32, data: [0, 1, null, 3, 4, 4]});
    const b  = Series.new({type: new Float32, data: [0, 1, 3, 5, 5, null]});
    const c  = Series.new({type: new Float32, data: [1, 2, 3, null, 5, 6]});
    const df = new DataFrame({'a': a, 'b': b, 'c': c});

    const result = df.dropNulls(1, df.numRows);

    // all columns are dropped as each one contains atleast one null value
    expect(result.numColumns).toEqual(0);
    expect(result.names).toEqual([]);
  });

test(
  'dataframe.dropNaNs(axis=0, thresh=df.numColumns), drop row with non-NaN values < numColumn (drop row if atleast one NaN)',
  () => {
    const a  = Series.new({type: new Float32, data: [0, 1, 2, 3, 4, 4]});
    const d  = Series.new({type: new Float32, data: [0, 1, 2, 3, 4, 4]});
    const b  = Series.new({type: new Float32, data: [0, NaN, 3, 5, 5, 6]});
    const c  = Series.new({type: new Float32, data: [NaN, NaN, NaN, NaN, NaN, NaN]});
    const df = new DataFrame({'a': a, 'b': b, 'c': c, 'd': d});

    // all rows are dropped, since every row contains atleast one NaN value
    const result = df.dropNaNs(0, df.numColumns);
    expect(result.numRows).toEqual(0);
  });

test('dataframe.dropNaNs(axis=0, thresh=1), drop row with non-NaN values < 1 (drop row if all NaN)',
     () => {
       const a  = Series.new({type: new Float32, data: [0, NaN, 2, 3, 4, 4]});
       const d  = Series.new({type: new Float32, data: [0, NaN, 2, 3, 4, 4]});
       const b  = Series.new({type: new Float32, data: [0, NaN, 3, 5, 5, 6]});
       const c  = Series.new({type: new Float32, data: [NaN, NaN, NaN, NaN, NaN, NaN]});
       const df = new DataFrame({'a': a, 'b': b, 'c': c, 'd': d});

       const expected_a = Series.new({type: new Float32, data: [0, 2, 3, 4, 4]});

       // row 1 is dropped as it contains all NaNs
       const result = df.dropNaNs(0, 1);
       const ra     = result.get('a');

       expect([...ra]).toEqual([...expected_a]);
       expect(result.numRows).toEqual(5);
     });

test('dataframe.dropNaNs(axis=1, thresh=1), drop columns with non-NaN values < 1 (drop if all NaN)',
     () => {
       const a  = Series.new({type: new Float32, data: [0, NaN, 2, 3, 4, 4]});
       const b  = Series.new({type: new Float32, data: [0, NaN, 3, 5, 5, 6]});
       const c  = Series.new({type: new Float32, data: [NaN, NaN, NaN, NaN, NaN, NaN]});
       const d  = Series.new({type: new Float32, data: [0, NaN, 2, 3, 4, 4]});
       const df = new DataFrame({'a': a, 'b': b, 'c': c, 'd': d});

       const result = df.dropNaNs(1, 1);

       // column c is dropped as it contains all NaN values
       expect(result.numColumns).toEqual(3);
       expect(result.names).toEqual(['a', 'b', 'd']);
     });

test(
  'dataframe.dropNaNs(axis=1, thresh=df.numRows), drop columns with non-NaN values < numRows (drop if atleast one NaN)',
  () => {
    const a  = Series.new({type: new Float32, data: [0, NaN, 2, 3, 4, 4]});
    const b  = Series.new({type: new Float32, data: [0, NaN, 3, 5, 5, 6]});
    const c  = Series.new({type: new Float32, data: [NaN, NaN, NaN, NaN, NaN, NaN]});
    const d  = Series.new({type: new Float32, data: [0, NaN, 2, 3, 4, 4]});
    const df = new DataFrame({'a': a, 'b': b, 'c': c, 'd': d});

    const result = df.dropNaNs(1, df.numRows);

    // all columns are dropped as each one contains atleast one null value
    expect(result.numColumns).toEqual(0);
    expect(result.names).toEqual([]);
  });

test('dataframe.cast', () => {
  const a  = Series.new({type: new Int32, data: [1, 2, 3, 4]});
  const b  = Series.new({type: new Float32, data: new Float32Buffer([1.5, 2.3, 3.1, 4])});
  const df = new DataFrame({'a': a, 'b': b});

  const result = df.cast({b: new Int32});

  expect(result.get('a').type).toBeInstanceOf(Int32);
  expect(result.get('b').type).toBeInstanceOf(Int32);
  expect([...result.get('b')]).toEqual([1, 2, 3, 4]);
});

test('dataframe.castAll', () => {
  const a  = Series.new({type: new Int8, data: [1, 2, 3, 4]});
  const b  = Series.new({type: new Float32, data: new Float32Buffer([1.5, 2.3, 3.1, 4])});
  const df = new DataFrame({'a': a, 'b': b});

  const result = df.castAll(new Int32);

  expect(result.get('a').type).toBeInstanceOf(Int32);
  expect(result.get('b').type).toBeInstanceOf(Int32);
  expect([...result.get('a')]).toEqual([1, 2, 3, 4]);
  expect([...result.get('b')]).toEqual([1, 2, 3, 4]);
});

describe('dataframe unaryops', () => {
  const a  = Series.new({type: new Int32, data: [-3, 0, 3]});
  const b  = Series.new({type: new Int8, data: [-3, 0, 3]});
  const c  = Series.new({type: new Float32, data: [-2.7, 0, 3.1]});
  const df = new DataFrame({'a': a, 'b': b, 'c': c});

  const d = Series.new(['foo', 'bar', 'foo']);
  // If a unary op is performed on this df, the result will be `never`
  // due to the df containing a string series.
  // (unary ops only support numeric series)
  const non_numeric_df = new DataFrame({'a': a, 'b': b, 'c': c, 'd': d});

  test('dataframe.sin', () => {
    const result = df.sin();
    expect([...result.get('a')]).toEqual([0, 0, 0]);
    expect([...result.get('b')]).toEqual([0, 0, 0]);
    expect([...result.get('c')]).toEqual([...c.sin()]);
  });

  test('dataframe.cos', () => {
    const result = df.cos();
    expect([...result.get('a')]).toEqual([0, 1, 0]);
    expect([...result.get('b')]).toEqual([0, 1, 0]);
    expect([...result.get('c')]).toEqual([...c.cos()]);
  });

  test('dataframe.tan', () => {
    const result = df.tan();
    expect([...result.get('a')]).toEqual([0, 0, 0]);
    expect([...result.get('b')]).toEqual([0, 0, 0]);
    expect([...result.get('c')]).toEqual([...c.tan()]);
  });

  test('dataframe.asin', () => {
    const result = df.asin();
    expect([...result.get('a')]).toEqual([-2147483648, 0, -2147483648]);
    expect([...result.get('b')]).toEqual([0, 0, 0]);
    expect([...result.get('c')]).toEqual([...c.asin()]);
  });

  test('dataframe.acos', () => {
    const result = df.acos();
    expect([...result.get('a')]).toEqual([-2147483648, 1, -2147483648]);
    expect([...result.get('b')]).toEqual([0, 1, 0]);
    expect([...result.get('c')]).toEqual([...c.acos()]);
  });

  test('dataframe.atan', () => {
    const result = df.atan();
    expect([...result.get('a')]).toEqual([-1, 0, 1]);
    expect([...result.get('b')]).toEqual([-1, 0, 1]);
    expect([...result.get('c')]).toEqual([...c.atan()]);
  });

  test('dataframe.sinh', () => {
    const result = df.sinh();
    expect([...result.get('a')]).toEqual([-10, 0, 10]);
    expect([...result.get('b')]).toEqual([-10, 0, 10]);
    expect([...result.get('c')]).toEqual([...c.sinh()]);
  });

  test('dataframe.cosh', () => {
    const result = df.cosh();
    expect([...result.get('a')]).toEqual([10, 1, 10]);
    expect([...result.get('b')]).toEqual([10, 1, 10]);
    expect([...result.get('c')]).toEqual([...c.cosh()]);
  });

  test('dataframe.tanh', () => {
    const result = df.tanh();
    expect([...result.get('a')]).toEqual([0, 0, 0]);
    expect([...result.get('b')]).toEqual([0, 0, 0]);
    expect([...result.get('c')]).toEqual([...c.tanh()]);
  });

  test('dataframe.asinh', () => {
    const result = df.asinh();
    expect([...result.get('a')]).toEqual([-1, 0, 1]);
    expect([...result.get('b')]).toEqual([-1, 0, 1]);
    expect([...result.get('c')]).toEqual([...c.asinh()]);
  });

  test('dataframe.acosh', () => {
    const result = df.acosh();
    expect([...result.get('a')]).toEqual([-2147483648, -2147483648, 1]);
    expect([...result.get('b')]).toEqual([0, 0, 1]);
    expect([...result.get('c')]).toEqual([...c.acosh()]);
  });

  test('dataframe.atanh', () => {
    const result = df.atanh();
    expect([...result.get('a')]).toEqual([-2147483648, 0, -2147483648]);
    expect([...result.get('b')]).toEqual([0, 0, 0]);
    expect([...result.get('c')]).toEqual([...c.atanh()]);
  });

  test('dataframe.exp', () => {
    const result = df.exp();
    expect([...result.get('a')]).toEqual([0, 1, 20]);
    expect([...result.get('b')]).toEqual([0, 1, 20]);
    expect([...result.get('c')]).toEqual([...c.exp()]);
  });

  test('dataframe.log', () => {
    const result = df.log();
    expect([...result.get('a')]).toEqual([-2147483648, -2147483648, 1]);
    expect([...result.get('b')]).toEqual([0, 0, 1]);
    expect([...result.get('c')]).toEqual([...c.log()]);
  });

  test('dataframe.sqrt', () => {
    const result = df.sqrt();
    expect([...result.get('a')]).toEqual([-2147483648, 0, 1]);
    expect([...result.get('b')]).toEqual([0, 0, 1]);
    expect([...result.get('c')]).toEqual([...c.sqrt()]);
  });

  test('dataframe.cbrt', () => {
    const result = df.cbrt();
    expect([...result.get('a')]).toEqual([-1, 0, 1]);
    expect([...result.get('b')]).toEqual([-1, 0, 1]);
    expect([...result.get('c')]).toEqual([...c.cbrt()]);
  });

  test('dataframe.ceil', () => {
    const result = df.ceil();
    expect([...result.get('a')]).toEqual([-3, 0, 3]);
    expect([...result.get('b')]).toEqual([-3, 0, 3]);
    expect([...result.get('c')]).toEqual([...c.ceil()]);
  });

  test('dataframe.floor', () => {
    const result = df.floor();
    expect([...result.get('a')]).toEqual([-3, 0, 3]);
    expect([...result.get('b')]).toEqual([-3, 0, 3]);
    expect([...result.get('c')]).toEqual([...c.floor()]);
  });

  test('dataframe.abs', () => {
    const result = df.abs();
    expect([...result.get('a')]).toEqual([3, 0, 3]);
    expect([...result.get('b')]).toEqual([3, 0, 3]);
    expect([...result.get('c')]).toEqual([...c.abs()]);
  });

  test('dataframe.not', () => {
    const result = df.not();
    expect([...result.get('a')]).toEqual([-3, 0, 3].map((x) => !x));
    expect([...result.get('b')]).toEqual([-3, 0, 3].map((x) => !x));
    expect([...result.get('c')]).toEqual([...c.not()]);
  });
});

describe('dataframe.replaceNulls', () => {
  test('replace with scalar', () => {
    const a      = Series.new([1, 2, 3, null]);
    const b      = Series.new([null, null, 7, null]);
    const c      = Series.new([null, null, null, null]);
    const df     = new DataFrame({'a': a, 'b': b, 'c': c});
    const result = df.replaceNulls(4);
    expect([...result.get('a')]).toEqual([1, 2, 3, 4]);
    expect([...result.get('b')]).toEqual([4, 4, 7, 4]);
    expect([...result.get('c')]).toEqual([4, 4, 4, 4]);

    // compare with series.replaceNulls result
    expect([...result.get('a')]).toEqual([...a.replaceNulls(4)]);
    expect([...result.get('b')]).toEqual([...b.replaceNulls(4)]);
    expect([...result.get('c')]).toEqual([...c.replaceNulls(4)]);
  });

  test('replace with seriesmap', () => {
    const a      = Series.new([1, 2, 3, null]);
    const b      = Series.new(['foo', 'bar', null, null]);
    const c      = Series.new([null, false, null, true]);
    const df     = new DataFrame({'a': a, 'b': b, 'c': c});
    const result = df.replaceNulls({
      'a': Series.new([1, 2, 3, 4]),
      'b': Series.new(['foo', 'bar', 'foo', 'bar']),
      'c': Series.new([false, false, true, true])
    });
    expect([...result.get('a')]).toEqual([1, 2, 3, 4]);
    expect([...result.get('b')]).toEqual(['foo', 'bar', 'foo', 'bar']);
    expect([...result.get('c')]).toEqual([false, false, true, true]);

    // compare with series.replaceNulls result
    expect([...result.get('a')]).toEqual([...a.replaceNulls(Series.new([1, 2, 3, 4]))]);
    expect([...result.get('b')]).toEqual([...b.replaceNulls(
      Series.new(['foo', 'bar', 'foo', 'bar']))]);
    expect([...result.get('c')]).toEqual([...c.replaceNulls(
      Series.new([false, false, true, true]))]);
  });
});

test('dataframe.nansToNulls', () => {
  const a  = Series.new({type: new Int32, data: [0, 1, 2, 3, 4, 4]});
  const b  = Series.new({type: new Float32, data: new Float32Buffer([0, NaN, 3, 5, 5, 6])});
  const df = new DataFrame({'a': a, 'b': b});

  expect(df.get('b').nullCount).toEqual(0);

  const result = df.nansToNulls();

  expect(result.get('b').nullCount).toEqual(1);
});

test('dataframe.isNaN', () => {
  const a      = Series.new({type: new Int32, data: [0, null, 2, 3, null]});
  const b      = Series.new({type: new Float32, data: [NaN, 0, 3, NaN, null]});
  const c      = Series.new([null, null, 'foo', 'bar', '']);
  const df     = new DataFrame({'a': a, 'b': b, 'c': c});
  const result = df.isNaN();

  const expected_a = Series.new(a);
  const expected_b = Series.new([true, false, false, true, false]);
  const expected_c = Series.new(c);

  expect([...result.get('a')]).toEqual([...expected_a]);
  expect([...result.get('b')]).toEqual([...expected_b]);
  expect([...result.get('c')]).toEqual([...expected_c]);
});

test('dataframe.isNull', () => {
  const a      = Series.new([0, null, 2, 3, null]);
  const b      = Series.new([NaN, 0, 3, NaN, null]);
  const c      = Series.new([null, null, 'foo', 'bar', '']);
  const df     = new DataFrame({'a': a, 'b': b, 'c': c});
  const result = df.isNull();

  const expected_a = Series.new({type: new Bool8, data: [false, true, false, false, true]});
  const expected_b = Series.new({type: new Bool8, data: [false, false, false, false, true]});
  const expected_c = Series.new({type: new Bool8, data: [true, true, false, false, false]});

  expect([...result.get('a')]).toEqual([...expected_a]);
  expect([...result.get('b')]).toEqual([...expected_b]);
  expect([...result.get('c')]).toEqual([...expected_c]);
});

test('dataframe.isNotNaN', () => {
  const a      = Series.new({type: new Float32, data: [0, NaN, 2, NaN, null]});
  const b      = Series.new({type: new Int32, data: [0, null, 2, 3, null]});
  const c      = Series.new([null, null, 'foo', 'bar', '']);
  const df     = new DataFrame({'a': a, 'b': b, 'c': c});
  const result = df.isNotNaN();

  const expected_a = Series.new([true, false, true, false, true]);
  const expected_b = Series.new(b);
  const expected_c = Series.new(c);

  expect([...result.get('a')]).toEqual([...expected_a]);
  expect([...result.get('b')]).toEqual([...expected_b]);
  expect([...result.get('c')]).toEqual([...expected_c]);
});

test('dataframe.isNotNull', () => {
  const a      = Series.new([null, 1, 2, 3, null]);
  const b      = Series.new([NaN, 0, 3, NaN, null]);
  const c      = Series.new(['foo', 'bar', null, null, '']);
  const df     = new DataFrame({'a': a, 'b': b, 'c': c});
  const result = df.isNotNull();

  const expected_a = Series.new({type: new Bool8, data: [false, true, true, true, false]});
  const expected_b = Series.new({type: new Bool8, data: [true, true, true, true, false]});
  const expected_c = Series.new({type: new Bool8, data: [true, true, false, false, true]});

  expect([...result.get('a')]).toEqual([...expected_a]);
  expect([...result.get('b')]).toEqual([...expected_b]);
  expect([...result.get('c')]).toEqual([...expected_c]);
});

test.each`
keep       | nullsEqual | nullsFirst | data                                                    | expected
${'first'} | ${true}    | ${true}    | ${[[4, null, 1, null, 3, 4], [4, null, 5, null, 8, 4]]} | ${[[null, 1, 3, 4], [null, 5, 8, 4]]}
${'last'}  | ${true}    | ${true}    | ${[[4, null, 1, null, 3, 4], [4, null, 5, null, 8, 4]]} | ${[[null, 1, 3, 4], [null, 5, 8, 4]]}
${'none'}  | ${true}    | ${true}    | ${[[4, null, 1, null, 3, 4], [4, null, 5, null, 8, 4]]} | ${[[1, 3], [5, 8]]}
${'first'} | ${false}   | ${true}    | ${[[4, null, 1, null, 3, 4], [4, null, 5, null, 8, 4]]} | ${[[null, null, 1, 3, 4], [null, null, 5, 8, 4]]}
${'last'}  | ${false}   | ${true}    | ${[[4, null, 1, null, 3, 4], [4, null, 5, null, 8, 4]]} | ${[[null, null, 1, 3, 4], [null, null, 5, 8, 4]]}
${'none'}  | ${false}   | ${true}    | ${[[4, null, 1, null, 3, 4], [4, null, 5, null, 8, 4]]} | ${[[null, null, 1, 3], [null, null, 5, 8]]}
${'first'} | ${true}    | ${false}   | ${[[4, null, 1, null, 3, 4], [4, null, 5, null, 8, 4]]} | ${[[1, 3, 4, null], [5, 8, 4, null]]}
${'last'}  | ${true}    | ${false}   | ${[[4, null, 1, null, 3, 4], [4, null, 5, null, 8, 4]]} | ${[[1, 3, 4, null], [5, 8, 4, null]]}
${'none'}  | ${true}    | ${false}   | ${[[4, null, 1, null, 3, 4], [4, null, 5, null, 8, 4]]} | ${[[1, 3], [5, 8]]}
${'first'} | ${false}   | ${false}   | ${[[4, null, 1, null, 3, 4], [4, null, 5, null, 8, 4]]} | ${[[1, 3, 4, null, null], [5, 8, 4, null, null]]}
${'last'}  | ${false}   | ${false}   | ${[[4, null, 1, null, 3, 4], [4, null, 5, null, 8, 4]]} | ${[[1, 3, 4, null, null], [5, 8, 4, null, null]]}
${'none'}  | ${false}   | ${false}   | ${[[4, null, 1, null, 3, 4], [4, null, 5, null, 8, 4]]} | ${[[1, 3, null, null], [5, 8, null, null]]}
`('DataFrame.dropDuplicates($keep, $nullsEqual, $nullsFirst)', ({keep, nullsEqual, nullsFirst, data, expected}) => {
  const a      = Series.new(data[0]);
  const b      = Series.new(data[1]);
  const df = new DataFrame({a, b});
  const result = df.dropDuplicates(keep, nullsEqual, nullsFirst);
  expect([...result.get('a')]).toEqual(expected[0]);
  expect([...result.get('b')]).toEqual(expected[1]);
});

test(`DataFrame.dropDuplicates("first", true, true, ['a'])`, () => {
  const a      = Series.new([4, null, 1, null, 3, 4]);
  const b      = Series.new([2, null, 5, null, 8, 9]);
  const df     = new DataFrame({a, b});
  const result = df.dropDuplicates('first', true, true, ['a']);
  expect([...result.get('a')]).toEqual([null, 1, 3, 4]);
  expect([...result.get('b')]).toEqual([null, 5, 8, 2]);
});
