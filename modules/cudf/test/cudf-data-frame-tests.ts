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

import {Float32Buffer, Int32Buffer, setDefaultAllocator, Uint8Buffer} from '@nvidia/cuda';
import {
  Bool8,
  DataFrame,
  Float32,
  GroupByMultiple,
  GroupBySingle,
  Int32,
  NullOrder,
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
  const result = df.orderBy({'a': {ascending: true, null_order: NullOrder.BEFORE}});

  const expected = [5, 0, 4, 1, 3, 2];
  expect([...result.toArrow()]).toEqual([...Buffer.from(expected)]);
});

test('DataFrame.orderBy (descending, non-null)', () => {
  const col    = Series.new({type: new Int32(), data: new Int32Buffer([1, 3, 5, 4, 2, 0])});
  const df     = new DataFrame({'a': col});
  const result = df.orderBy({'a': {ascending: false, null_order: NullOrder.BEFORE}});

  const expected = [2, 3, 1, 4, 0, 5];
  expect([...result.toArrow()]).toEqual([...Buffer.from(expected)]);
});

test('DataFrame.orderBy (ascending, null before)', () => {
  const mask = new Uint8Buffer(BoolVector.from([1, 0, 1, 1, 1, 1]).values);
  const col =
    Series.new({type: new Int32(), data: new Int32Buffer([1, 3, 5, 4, 2, 0]), nullMask: mask});
  const df     = new DataFrame({'a': col});
  const result = df.orderBy({'a': {ascending: true, null_order: NullOrder.BEFORE}});

  const expected = [1, 5, 0, 4, 3, 2];
  expect([...result.toArrow()]).toEqual([...Buffer.from(expected)]);
});

test('DataFrame.orderBy (ascending, null after)', () => {
  const mask = new Uint8Buffer(BoolVector.from([1, 0, 1, 1, 1, 1]).values);
  const col =
    Series.new({type: new Int32(), data: new Int32Buffer([1, 3, 5, 4, 2, 0]), nullMask: mask});
  const df     = new DataFrame({'a': col});
  const result = df.orderBy({'a': {ascending: true, null_order: NullOrder.AFTER}});

  const expected = [5, 0, 4, 3, 2, 1];
  expect([...result.toArrow()]).toEqual([...Buffer.from(expected)]);
});

test('DataFrame.orderBy (descendng, null before)', () => {
  const mask = new Uint8Buffer(BoolVector.from([1, 0, 1, 1, 1, 1]).values);
  const col =
    Series.new({type: new Int32(), data: new Int32Buffer([1, 3, 5, 4, 2, 0]), nullMask: mask});
  const df     = new DataFrame({'a': col});
  const result = df.orderBy({'a': {ascending: false, null_order: NullOrder.BEFORE}});

  const expected = [2, 3, 4, 0, 5, 1];

  expect([...result.toArrow()]).toEqual([...Buffer.from(expected)]);
});

test('DataFrame.orderBy (descending, null after)', () => {
  const mask = new Uint8Buffer(BoolVector.from([1, 0, 1, 1, 1, 1]).values);
  const col =
    Series.new({type: new Int32(), data: new Int32Buffer([1, 3, 5, 4, 2, 0]), nullMask: mask});
  const df     = new DataFrame({'a': col});
  const result = df.orderBy({'a': {ascending: false, null_order: NullOrder.AFTER}});

  const expected = [1, 2, 3, 4, 0, 5];
  expect([...result.toArrow()]).toEqual([...Buffer.from(expected)]);
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
  expect([...ra.toArrow()]).toEqual([...expected_a.toArrow()]);

  const expected_b = Series.new({type: new Float32(), data: new Float32Buffer([2.0, 4.0, 5.0])});
  expect([...rb.toArrow()]).toEqual([...expected_b.toArrow()]);
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

test('Series.filter', () => {
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
  expect([...ra.toArrow()]).toEqual([...expected_a.toArrow()]);

  const expected_b = Series.new({type: new Float32(), data: new Float32Buffer([2.0, 4.0, 5.0])});
  expect([...rb.toArrow()]).toEqual([...expected_b.toArrow()]);
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

    expect([...ra.toArrow()]).toEqual([...expected_a.toArrow()]);
    expect([...rc.toArrow()]).toEqual([...expected_c.toArrow()]);
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

       expect([...ra.toArrow()]).toEqual([...expected_a.toArrow()]);
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

test('dataframe.nansToNulls', () => {
  const a  = Series.new({type: new Int32, data: [0, 1, 2, 3, 4, 4]});
  const b  = Series.new({type: new Float32, data: new Float32Buffer([0, NaN, 3, 5, 5, 6])});
  const df = new DataFrame({'a': a, 'b': b});

  expect(df.get('b').nullCount).toEqual(0);

  const result = df.nansToNulls();

  expect(result.get('b').nullCount).toEqual(1);
});
