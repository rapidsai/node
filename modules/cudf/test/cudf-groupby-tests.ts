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

/* eslint-disable @typescript-eslint/no-non-null-assertion */

import './jest-extensions';

import {setDefaultAllocator} from '@rapidsai/cuda';
import {
  DataFrame,
  DataType,
  Float64,
  GroupByMultiple,
  GroupBySingle,
  Int32,
  Series,
  // StructSeries
} from '@rapidsai/cudf';
import {CudaMemoryResource, DeviceBuffer} from '@rapidsai/rmm';
import * as arrow from 'apache-arrow';

const mr = new CudaMemoryResource();

setDefaultAllocator((byteLength: number) => new DeviceBuffer(byteLength, mr));

function makeBasicData(values: number[]) {
  const a = Series.new({type: new Int32, data: [1, 2, 3, 1, 2, 2, 1, 3, 3, 2]});
  const b = Series.new({type: new Float64, data: values});
  const c = Series.new({type: new Float64, data: values});
  return new DataFrame({a, b, 'c': c});
}

function basicAggCompare<T extends {a: DataType, b: DataType, c: DataType}, E extends any[]>(
  result: DataFrame<T>, expected: E): void {
  const ra = result.get('a');
  const rb = result.get('b');
  const rc = result.get('c');

  const a_expected = Series.new({type: new Int32, data: [1, 2, 3]});
  expect([...ra]).toEqual([...a_expected]);

  const b_expected = Series.new({type: rb.type, data: expected});

  expect(rb.type).toEqual(b_expected.type);
  expect(rc.type).toEqual(b_expected.type);

  if (arrow.DataType.isList(rb.type)) {
    const isIterable = (x: any): x is Iterable<any> =>  //
      (x && typeof x === 'object' && typeof x[Symbol.iterator]);
    const unwrap = (xs: any[]): any[][] =>              //
      xs.map((ys) => isIterable(ys) ? unwrap([...ys]) : ys);

    const actual_rb_lists  = unwrap([...rb]);
    const actual_rc_lists  = unwrap([...rc]);
    const b_expected_lists = unwrap([...b_expected]);

    expect(actual_rb_lists).toEqual(b_expected_lists);
    expect(actual_rc_lists).toEqual(b_expected_lists);
  } else {
    expect(rb.toArray()).toEqualTypedArray(b_expected.toArray() as any);
    expect(rc.toArray()).toEqualTypedArray(b_expected.toArray() as any);
  }
}

test('getGroups basic', () => {
  const a  = Series.new({type: new Int32, data: [1, 1, 2, 1, 2, 3]});
  const df = new DataFrame({'a': a});

  const grp = new GroupBySingle(df, {by: 'a'});

  const groups = grp.getGroups();

  const keys_result = groups['keys'].get('a');
  expect([...keys_result]).toEqual([1, 1, 1, 2, 2, 3]);

  expect(groups.values).toBeUndefined();

  expect([...groups['offsets']]).toEqual([0, 3, 5, 6]);
});

test('getGroups basic two columns', () => {
  const a  = Series.new({type: new Int32, data: [1, 1, 2, 1, 2, 3]});
  const aa = Series.new({type: new Int32, data: [4, 5, 4, 4, 4, 3]});
  const df = new DataFrame({'a': a, 'aa': aa});

  const grp = new GroupByMultiple(df, {by: ['a', 'aa'], index_key: 'out'});

  const groups = grp.getGroups();

  const keys_result_a = groups['keys'].get('a');
  expect([...keys_result_a]).toEqual([1, 1, 1, 2, 2, 3]);

  const keys_result_aa = groups['keys'].get('aa');
  expect([...keys_result_aa]).toEqual([4, 4, 5, 4, 4, 3]);

  expect(groups.values).toBeUndefined();

  expect([...groups['offsets']]).toEqual([0, 2, 3, 5, 6]);
});

test('getGroups empty', () => {
  const a  = Series.new({type: new Int32, data: []});
  const df = new DataFrame({'a': a});

  const grp = new GroupBySingle(df, {by: 'a'});

  const groups = grp.getGroups();

  const keys_result = groups['keys'].get('a');
  expect(keys_result.length).toBe(0);

  expect(groups.values).toBeUndefined();

  expect([...groups['offsets']]).toEqual([0]);
});

test('getGroups basic with values', () => {
  const a  = Series.new({type: new Int32, data: [5, 4, 3, 2, 1, 0]});
  const b  = Series.new({type: new Int32, data: [0, 0, 1, 1, 2, 2]});
  const c  = Series.new({type: new Int32, data: [0, 0, 1, 1, 2, 2]});
  const df = new DataFrame({a, b, c});

  const grp = new GroupBySingle(df, {by: 'a'});

  const groups = grp.getGroups();

  const keys_result = groups['keys'].get('a');
  expect([...keys_result]).toEqual([0, 1, 2, 3, 4, 5]);

  // eslint-disable-next-line @typescript-eslint/no-non-null-asserted-optional-chain
  const values_result_b = groups.values?.get('b')!;
  expect(values_result_b).toBeDefined();
  expect([...values_result_b]).toEqual([2, 2, 1, 1, 0, 0]);

  // eslint-disable-next-line @typescript-eslint/no-non-null-asserted-optional-chain
  const values_result_c = groups.values?.get('c')!;
  expect(values_result_c).toBeDefined();
  expect([...values_result_c]).toEqual([2, 2, 1, 1, 0, 0]);

  expect([...groups['offsets']]).toEqual([0, 1, 2, 3, 4, 5, 6]);
});

test('getGroups basic two columns with values', () => {
  const a  = Series.new({type: new Int32, data: [5, 4, 3, 2, 1, 0]});
  const aa = Series.new({type: new Int32, data: [4, 5, 4, 4, 4, 3]});
  const b  = Series.new({type: new Int32, data: [0, 0, 1, 1, 2, 2]});
  const c  = Series.new({type: new Int32, data: [0, 0, 1, 1, 2, 2]});
  const df = new DataFrame({a, aa, b, c});

  const grp = new GroupByMultiple(df, {by: ['a', 'aa'], index_key: 'out'});

  const groups = grp.getGroups();

  const keys_result_a = groups['keys'].get('a');
  expect([...keys_result_a]).toEqual([0, 1, 2, 3, 4, 5]);

  const keys_result_aa = groups['keys'].get('aa');
  expect([...keys_result_aa]).toEqual([3, 4, 4, 4, 5, 4]);

  // eslint-disable-next-line @typescript-eslint/no-non-null-asserted-optional-chain
  const values_result_b = groups.values?.get('b')!;
  expect(values_result_b).toBeDefined();
  expect([...values_result_b]).toEqual([2, 2, 1, 1, 0, 0]);

  // eslint-disable-next-line @typescript-eslint/no-non-null-asserted-optional-chain
  const values_result_c = groups.values?.get('c')!;
  expect(values_result_c).toBeDefined();
  expect([...values_result_c]).toEqual([2, 2, 1, 1, 0, 0]);

  expect([...groups['offsets']]).toEqual([0, 1, 2, 3, 4, 5, 6]);
});

test('getGroups all nulls', () => {
  const a  = Series.new({type: new Int32, data: [null, null, null, null, null, null]});
  const df = new DataFrame({'a': a});

  const grp = new GroupBySingle(df, {by: 'a'});

  const groups = grp.getGroups();

  const keys_result = groups['keys'].get('a');
  expect(keys_result.length).toBe(0);

  expect(groups.values).toBeUndefined();

  expect([...groups['offsets']]).toEqual([0]);
});

test('getGroups some nulls', () => {
  const a  = Series.new({type: new Int32, data: [1, null, 3, null, null, 2]});
  const b  = Series.new({type: new Int32, data: [1, 2, 3, 4, 5, 6]});
  const c  = Series.new({type: new Int32, data: [1, 2, 3, 4, 5, 6]});
  const df = new DataFrame({a, b, c});

  const grp = new GroupBySingle(df, {by: 'a'});

  const groups = grp.getGroups();

  const keys_result = groups['keys'].get('a');
  expect([...keys_result]).toEqual([1, 2, 3]);
  expect(keys_result.nullCount).toBe(0);

  // eslint-disable-next-line @typescript-eslint/no-non-null-asserted-optional-chain
  const values_result_b = groups.values?.get('b')!;
  expect(values_result_b).toBeDefined();
  expect([...values_result_b]).toEqual([1, 6, 3]);

  // eslint-disable-next-line @typescript-eslint/no-non-null-asserted-optional-chain
  const values_result_c = groups.values?.get('c')!;
  expect(values_result_c).toBeDefined();
  expect([...values_result_c]).toEqual([1, 6, 3]);

  expect([...groups['offsets']]).toEqual([0, 1, 2, 3]);
});

test('aggregation column name with two columns', () => {
  const a  = Series.new({type: new Int32, data: [5, 4, 3, 2, 1, 0]});
  const aa = Series.new({type: new Int32, data: [4, 5, 4, 4, 4, 3]});
  const b  = Series.new({type: new Int32, data: [0, 0, 1, 1, 2, 2]});
  const df = new DataFrame({'a': a, 'aa': aa, 'b': b});

  const grp = new GroupByMultiple(df, {by: ['a', 'aa'], index_key: 'out'});

  const agg = grp.max();

  const result_out = agg.get('out');

  const keys_result_a  = result_out.getChild('a');
  const keys_result_aa = result_out.getChild('aa');

  const sorter = [0, 1, 2, 3, 4, 5];
  const ka     = [...keys_result_a];
  sorter.sort((i, j) => ka[i]! - ka[j]!);

  const sorted_a =
    keys_result_a.gather(Series.new({type: new Int32, data: new Int32Array(sorter)}));
  expect([...sorted_a]).toEqual([0, 1, 2, 3, 4, 5]);

  const sorted_aa =
    keys_result_aa.gather(Series.new({type: new Int32, data: new Int32Array(sorter)}));
  expect([...sorted_aa]).toEqual([3, 4, 4, 4, 5, 4]);

  const sorted_b = agg.get('b').gather(Series.new({type: new Int32, data: new Int32Array(sorter)}));
  expect([...sorted_b]).toEqual([2, 2, 1, 1, 0, 0]);
});

test('aggregation existing column name with two columns raises', () => {
  const a  = Series.new({type: new Int32, data: [5, 4, 3, 2, 1, 0]});
  const aa = Series.new({type: new Int32, data: [4, 5, 4, 4, 4, 3]});
  const b  = Series.new({type: new Int32, data: [0, 0, 1, 1, 2, 2]});
  const df = new DataFrame({'a': a, 'aa': aa, 'b': b});

  const grp = new GroupByMultiple(df, {by: ['a', 'aa'], index_key: 'b'});

  expect(() => grp.max()).toThrowError();
});

test('Groupby argmax basic', () => {
  const df  = makeBasicData([9, 8, 7, 6, 5, 4, 3, 2, 1, 0]);
  const grp = new GroupBySingle(df, {by: 'a'});
  basicAggCompare(grp.argmax(), [0, 1, 2]);
});

test('Groupby argmin basic', () => {
  const df  = makeBasicData([9, 8, 7, 6, 5, 4, 3, 2, 1, 0]);
  const grp = new GroupBySingle(df, {by: 'a'});
  basicAggCompare(grp.argmin(), [6, 9, 8]);
});

test('Groupby count basic', () => {
  const df  = makeBasicData([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  const grp = new GroupBySingle(df, {by: 'a'});
  basicAggCompare(grp.count(), [3, 4, 3]);
});

test('Groupby max basic', () => {
  const df  = makeBasicData([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  const grp = new GroupBySingle(df, {by: 'a'});
  basicAggCompare(grp.max(), [6, 9, 8]);
});

test('Groupby mean basic', () => {
  const df  = makeBasicData([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  const grp = new GroupBySingle(df, {by: 'a'});
  basicAggCompare(grp.mean(), [3, 19 / 4, 17 / 3]);
});

test('Groupby median basic', () => {
  const df  = makeBasicData([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  const grp = new GroupBySingle(df, {by: 'a'});
  basicAggCompare(grp.median(), [3, 4.5, 7]);
});

test('Groupby min basic', () => {
  const df  = makeBasicData([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  const grp = new GroupBySingle(df, {by: 'a'});
  basicAggCompare(grp.min(), [0, 1, 2]);
});

test('Groupby nth basic', () => {
  const a   = Series.new({type: new Int32, data: [1, 1, 1, 2, 2, 2, 3, 3, 3]});
  const b   = Series.new({type: new Float64, data: [1, 2, 3, 10, 20, 30, 100, 200, 300]});
  const c   = Series.new({type: new Float64, data: [1, 2, 3, 10, 20, 30, 100, 200, 300]});
  const df  = new DataFrame({a, b, c});
  const grp = new GroupBySingle(df, {by: 'a'});
  basicAggCompare(grp.nth(0), [1, 10, 100]);
  basicAggCompare(grp.nth(1), [2, 20, 200]);
  basicAggCompare(grp.nth(2), [3, 30, 300]);
});

test('Groupby nth uneven', () => {
  const a      = Series.new({type: new Int32, data: [1, 1, 1, 2, 2, 2, 3, 3]});
  const b      = Series.new({type: new Float64, data: [1, 2, 3, 10, 20, 30, 100, 200]});
  const c      = Series.new({type: new Float64, data: [1, 2, 3, 10, 20, 30, 100, 200]});
  const df     = new DataFrame({a, b, c});
  const grp    = new GroupBySingle(df, {by: 'a'});
  const result = grp.nth(2);
  basicAggCompare(result, [3, 30, 0]);
  expect(result.get('b').nullCount).toBe(1);
});

test('Groupby nunique basic', () => {
  const df  = makeBasicData([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  const grp = new GroupBySingle(df, {by: 'a'});
  basicAggCompare(grp.nunique(), [3, 4, 3]);
});

test('Groupby quantile uneven', () => {
  const a      = Series.new({type: new Int32, data: [1, 2, 3, 1, 2, 2, 1, 3, 3, 2]});
  const b      = Series.new({type: new Float64, data: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]});
  const c      = Series.new({type: new Float64, data: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]});
  const df     = new DataFrame({a, b, c});
  const grp    = new GroupBySingle(df, {by: 'a'});
  const result = grp.quantile(0.5);
  basicAggCompare(result, [3., 4.5, 7.]);
  expect(result.get('b').nullCount).toBe(0);
});

test('Groupby std basic', () => {
  const df  = makeBasicData([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  const grp = new GroupBySingle(df, {by: 'a'});
  basicAggCompare(grp.std(), [3, Math.sqrt(131 / 12), Math.sqrt(31 / 3)]);
});

test('Groupby sum basic', () => {
  const df  = makeBasicData([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  const grp = new GroupBySingle(df, {by: 'a'});
  basicAggCompare(grp.sum(), [9, 19, 17]);
});

test('Groupby var basic', () => {
  const df  = makeBasicData([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  const grp = new GroupBySingle(df, {by: 'a'});
  basicAggCompare(grp.var(), [9, 131 / 12, 31 / 3]);
});

test('Groupby collectList basic', () => {
  // keys=[[1, 1, 1], [2, 2, 2, 2], [3, 3, 3]]
  // vals=[[0, 3, 6], [1, 4, 5, 9], [2, 7, 8]]
  const df  = makeBasicData([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  const grp = new GroupBySingle(df, {by: 'a'});
  basicAggCompare(grp.collectList(), [[0, 3, 6], [1, 4, 5, 9], [2, 7, 8]]);
});

test('Groupby collectSet all unique', () => {
  const df  = makeBasicData([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  const grp = new GroupBySingle(df, {by: 'a'});
  basicAggCompare(grp.collectSet(), [[0, 3, 6], [1, 4, 5, 9], [2, 7, 8]]);
});

test('Groupby collectSet with duplicates', () => {
  const df  = makeBasicData([0, 1, 2, 0, 1, 1, 6, 2, 8, 9]);
  const grp = new GroupBySingle(df, {by: 'a'});
  basicAggCompare(grp.collectSet(), [[0, 6], [1, 9], [2, 8]]);
});

export type BasicAggType =
  'sum'|'min'|'max'|'argmin'|'argmax'|'mean'|'count'|'nunique'|'var'|'std'|'median';

const BASIC_AGGS: BasicAggType[] =
  ['sum', 'min', 'max', 'argmin', 'argmax', 'mean', 'count', 'nunique', 'var', 'std', 'median'];

for (const agg of BASIC_AGGS) {
  test(`Groupby ${agg} empty`, () => {
    const a      = Series.new({type: new Int32, data: []});
    const b      = Series.new({type: new Float64, data: []});
    const c      = Series.new({type: new Float64, data: []});
    const df     = new DataFrame({a, b, c});
    const grp    = new GroupBySingle(df, {by: 'a'});
    const result = grp[agg]();
    expect(result.get('a').length).toBe(0);
    expect(result.get('b').length).toBe(0);
    expect(result.get('c').length).toBe(0);
  });
}

test(`Groupby nth empty`, () => {
  const a      = Series.new({type: new Int32, data: []});
  const b      = Series.new({type: new Float64, data: []});
  const c      = Series.new({type: new Float64, data: []});
  const df     = new DataFrame({a, b, c});
  const grp    = new GroupBySingle(df, {by: 'a'});
  const result = grp.nth(0);
  expect(result.get('a').length).toBe(0);
  expect(result.get('b').length).toBe(0);
  expect(result.get('c').length).toBe(0);
});

test(`Groupby quantile empty`, () => {
  const a      = Series.new({type: new Int32, data: []});
  const b      = Series.new({type: new Float64, data: []});
  const c      = Series.new({type: new Float64, data: []});
  const df     = new DataFrame({a, b, c});
  const grp    = new GroupBySingle(df, {by: 'a'});
  const result = grp.quantile(0.5);
  expect(result.get('a').length).toBe(0);
  expect(result.get('b').length).toBe(0);
  expect(result.get('c').length).toBe(0);
});

for (const agg of BASIC_AGGS) {
  test(`Groupby ${agg} null keys`, () => {
    const a      = Series.new({type: new Int32, data: [null, null, null]});
    const b      = Series.new({type: new Float64, data: [3, 4, 5]});
    const c      = Series.new({type: new Float64, data: [3, 4, 5]});
    const df     = new DataFrame({a, b, c});
    const grp    = new GroupBySingle(df, {by: 'a'});
    const result = grp[agg]();
    expect(result.get('a').length).toBe(0);
    expect(result.get('b').length).toBe(0);
    expect(result.get('c').length).toBe(0);
  });
}

test(`Groupby nth null keys`, () => {
  const a      = Series.new({type: new Int32, data: [null, null, null]});
  const b      = Series.new({type: new Float64, data: [3, 4, 5]});
  const c      = Series.new({type: new Float64, data: [3, 4, 5]});
  const df     = new DataFrame({a, b, c});
  const grp    = new GroupBySingle(df, {by: 'a'});
  const result = grp.nth(0);
  expect(result.get('a').length).toBe(0);
  expect(result.get('b').length).toBe(0);
  expect(result.get('c').length).toBe(0);
});

test(`Groupby quantile null keys`, () => {
  const a      = Series.new({type: new Int32, data: [null, null, null]});
  const b      = Series.new({type: new Float64, data: [3, 4, 5]});
  const c      = Series.new({type: new Float64, data: [3, 4, 5]});
  const df     = new DataFrame({a, b, c});
  const grp    = new GroupBySingle(df, {by: 'a'});
  const result = grp.quantile(0.5);
  expect(result.get('a').length).toBe(0);
  expect(result.get('b').length).toBe(0);
  expect(result.get('c').length).toBe(0);
});

for (const agg of BASIC_AGGS) {
  test(`Groupby ${agg} null values`, () => {
    const a      = Series.new({type: new Int32, data: [1, 1, 1]});
    const b      = Series.new({type: new Float64, data: [null, null, null]});
    const c      = Series.new({type: new Float64, data: [null, null, null]});
    const df     = new DataFrame({a, b, c});
    const grp    = new GroupBySingle(df, {by: 'a'});
    const result = grp[agg]();
    expect([...result.get('a')]).toEqual([1]);
    expect(result.get('a').nullCount).toBe(0);
    expect(result.get('b').length).toBe(1);
    expect(result.get('c').length).toBe(1);
    if (agg == 'count' || agg == 'nunique') {
      expect(result.get('b').nullCount).toBe(0);
      expect(result.get('c').nullCount).toBe(0);
    } else {
      expect(result.get('b').nullCount).toBe(1);
      expect(result.get('c').nullCount).toBe(1);
    }
  });
}

test(`Groupby nth null values`, () => {
  const a      = Series.new({type: new Int32, data: [1, 1, 1]});
  const b      = Series.new({type: new Float64, data: [null, null, null]});
  const c      = Series.new({type: new Float64, data: [null, null, null]});
  const df     = new DataFrame({a, b, c});
  const grp    = new GroupBySingle(df, {by: 'a'});
  const result = grp.nth(0);
  expect([...result.get('a')]).toEqual([1]);
  expect(result.get('a').nullCount).toBe(0);
  expect(result.get('b').length).toBe(1);
  expect(result.get('b').nullCount).toBe(1);
  expect(result.get('c').length).toBe(1);
  expect(result.get('c').nullCount).toBe(1);
});

test(`Groupby quantile null values`, () => {
  const a      = Series.new({type: new Int32, data: [1, 1, 1]});
  const b      = Series.new({type: new Float64, data: [null, null, null]});
  const c      = Series.new({type: new Float64, data: [null, null, null]});
  const df     = new DataFrame({a, b, c});
  const grp    = new GroupBySingle(df, {by: 'a'});
  const result = grp.quantile(0.5);
  expect([...result.get('a')]).toEqual([1]);
  expect(result.get('a').nullCount).toBe(0);
  expect(result.get('b').length).toBe(1);
  expect(result.get('b').nullCount).toBe(1);
  expect(result.get('c').length).toBe(1);
  expect(result.get('c').nullCount).toBe(1);
});
