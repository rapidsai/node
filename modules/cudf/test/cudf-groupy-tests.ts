/* eslint-disable @typescript-eslint/no-non-null-assertion */
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

import './jest-extensions'

import {setDefaultAllocator} from '@nvidia/cuda';
import {DataFrame, DataType, Float64, GroupBy, Int32, Series} from '@nvidia/cudf';
import {CudaMemoryResource, DeviceBuffer} from '@nvidia/rmm';

const mr = new CudaMemoryResource();

setDefaultAllocator((byteLength: number) => new DeviceBuffer(byteLength, mr));

function makeBasicData(values: number[]): DataFrame {
  const a = Series.new({type: new Int32, data: [1, 2, 3, 1, 2, 2, 1, 3, 3, 2]});
  const b = Series.new({type: new Float64, data: values});
  return new DataFrame({'a': a, 'b': b});
}

function basicAggCompare<T extends {a: DataType, b: DataType}>(result: DataFrame<T>,
                                                               expected: number[]): void {
  const ra = result.get('a');
  const rb = result.get('b');

  const a_expected = Series.new({type: new Int32, data: [1, 2, 3]});
  expect([...ra.toArrow()]).toEqual([...a_expected.toArrow()]);

  const b_expected = Series.new({type: rb.type, data: expected});
  expect(rb.toArrow().toArray()).toEqualTypedArray(b_expected.toArrow().toArray() as any);
}

test('Groupby basic', () => {
  const a  = Series.new({type: new Int32, data: [1, 1, 2, 1, 2, 3]});
  const df = new DataFrame({'a': a});

  const grp = new GroupBy({obj: df, by: ['a']});

  const groups = grp.getGroups();

  const keys_result = groups['keys'].get('a');
  expect([...keys_result.toArrow()]).toEqual([1, 1, 1, 2, 2, 3]);

  expect(groups.values).toBeUndefined();

  expect([...groups['offsets']]).toEqual([0, 3, 5, 6]);
});

test('Groupby basic two columns', () => {
  const a  = Series.new({type: new Int32, data: [1, 1, 2, 1, 2, 3]});
  const aa = Series.new({type: new Int32, data: [4, 5, 4, 4, 4, 3]});
  const df = new DataFrame({'a': a, 'aa': aa});

  const grp = new GroupBy({obj: df, by: ['a', 'aa']});

  const groups = grp.getGroups();

  const keys_result_a = groups['keys'].get('a');
  expect([...keys_result_a.toArrow()]).toEqual([1, 1, 1, 2, 2, 3]);

  const keys_result_aa = groups['keys'].get('aa');
  expect([...keys_result_aa.toArrow()]).toEqual([4, 4, 5, 4, 4, 3]);

  expect(groups.values).toBeUndefined();

  expect([...groups['offsets']]).toEqual([0, 2, 3, 5, 6]);
});

test('Groupby empty', () => {
  const a  = Series.new({type: new Int32, data: []});
  const df = new DataFrame({'a': a});

  const grp = new GroupBy({obj: df, by: ['a']});

  const groups = grp.getGroups();

  const keys_result = groups['keys'].get('a');
  expect(keys_result.length).toBe(0);

  expect(groups.values).toBeUndefined();

  expect([...groups['offsets']]).toEqual([0]);
});

test('Groupby basic with values', () => {
  const a  = Series.new({type: new Int32, data: [5, 4, 3, 2, 1, 0]});
  const b  = Series.new({type: new Int32, data: [0, 0, 1, 1, 2, 2]});
  const df = new DataFrame({'a': a, 'b': b});

  const grp = new GroupBy({obj: df, by: ['a']});

  const groups = grp.getGroups();

  const keys_result = groups['keys'].get('a');
  expect([...keys_result.toArrow()]).toEqual([0, 1, 2, 3, 4, 5]);

  // eslint-disable-next-line @typescript-eslint/no-non-null-asserted-optional-chain
  const values_result = groups.values?.get('b')!;
  expect(values_result).toBeDefined()
  expect([...values_result.toArrow()]).toEqual([2, 2, 1, 1, 0, 0]);

  expect([...groups['offsets']]).toEqual([0, 1, 2, 3, 4, 5, 6]);
});

test('Groupby basic two columns with values', () => {
  const a  = Series.new({type: new Int32, data: [5, 4, 3, 2, 1, 0]});
  const aa = Series.new({type: new Int32, data: [4, 5, 4, 4, 4, 3]});
  const b  = Series.new({type: new Int32, data: [0, 0, 1, 1, 2, 2]});
  const df = new DataFrame({'a': a, 'aa': aa, 'b': b});

  const grp = new GroupBy({obj: df, by: ['a', 'aa']});

  const groups = grp.getGroups();

  const keys_result_a = groups['keys'].get('a');
  expect([...keys_result_a.toArrow()]).toEqual([0, 1, 2, 3, 4, 5]);

  const keys_result_aa = groups['keys'].get('aa');
  expect([...keys_result_aa.toArrow()]).toEqual([3, 4, 4, 4, 5, 4]);

  // eslint-disable-next-line @typescript-eslint/no-non-null-asserted-optional-chain
  const values_result = groups.values?.get('b')!;
  expect(values_result).toBeDefined()
  expect([...values_result.toArrow()]).toEqual([2, 2, 1, 1, 0, 0]);

  expect([...groups['offsets']]).toEqual([0, 1, 2, 3, 4, 5, 6]);
});

test('Groupby all nulls', () => {
  const a  = Series.new({
    type: new Int32,
    data: [1, 1, 2, 3, 1, 2],
    nullMask: [false, false, false, false, false, false],
  });
  const df = new DataFrame({'a': a});

  const grp = new GroupBy({obj: df, by: ['a']});

  const groups = grp.getGroups();

  const keys_result = groups['keys'].get('a');
  expect(keys_result.length).toBe(0);

  expect(groups.values).toBeUndefined();

  expect([...groups['offsets']]).toEqual([0]);
});

test('Groupby some nulls', () => {
  const a  = Series.new({
    type: new Int32,
    data: [1, 1, 3, 2, 1, 2],
    nullMask: [true, false, true, false, false, true],
  });
  const b  = Series.new({type: new Int32, data: [1, 2, 3, 4, 5, 6]});
  const df = new DataFrame({'a': a, 'b': b});

  const grp = new GroupBy({obj: df, by: ['a']});

  const groups = grp.getGroups();

  const keys_result = groups['keys'].get('a');
  expect([...keys_result.toArrow()]).toEqual([1, 2, 3]);
  expect(keys_result.nullCount).toBe(0)

  // eslint-disable-next-line @typescript-eslint/no-non-null-asserted-optional-chain
  const values_result = groups.values?.get('b')!;
  expect(values_result).toBeDefined()
  expect([...values_result.toArrow()]).toEqual([1, 6, 3]);

  expect([...groups['offsets']]).toEqual([0, 1, 2, 3]);
});

test('Groupby argmax basic', () => {
  const df  = makeBasicData([9, 8, 7, 6, 5, 4, 3, 2, 1, 0]);
  const grp = new GroupBy({obj: df, by: ['a']});
  basicAggCompare(grp.argmax(), [0, 1, 2]);
});

test('Groupby argmin basic', () => {
  const df  = makeBasicData([9, 8, 7, 6, 5, 4, 3, 2, 1, 0]);
  const grp = new GroupBy({obj: df, by: ['a']});
  basicAggCompare(grp.argmin(), [6, 9, 8]);
});

test('Groupby count basic', () => {
  const df  = makeBasicData([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  const grp = new GroupBy({obj: df, by: ['a']});
  basicAggCompare(grp.count(), [3, 4, 3]);
});

test('Groupby max basic', () => {
  const df  = makeBasicData([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  const grp = new GroupBy({obj: df, by: ['a']});
  basicAggCompare(grp.max(), [6, 9, 8]);
});

test('Groupby mean basic', () => {
  const df  = makeBasicData([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  const grp = new GroupBy({obj: df, by: ['a']});
  basicAggCompare(grp.mean(), [3, 19 / 4, 17 / 3]);
});

test('Groupby median basic', () => {
  const df  = makeBasicData([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  const grp = new GroupBy({obj: df, by: ['a']});
  basicAggCompare(grp.median(), [3, 4.5, 7]);
});

test('Groupby min basic', () => {
  const df  = makeBasicData([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  const grp = new GroupBy({obj: df, by: ['a']});
  basicAggCompare(grp.min(), [0, 1, 2]);
});

test('Groupby nth basic', () => {
  const a   = Series.new({type: new Int32, data: [1, 1, 1, 2, 2, 2, 3, 3, 3]});
  const b   = Series.new({type: new Float64, data: [1, 2, 3, 10, 20, 30, 100, 200, 300]});
  const df  = new DataFrame({'a': a, 'b': b});
  const grp = new GroupBy({obj: df, by: ['a']});
  basicAggCompare(grp.nth(0), [1, 10, 100]);
  basicAggCompare(grp.nth(1), [2, 20, 200]);
  basicAggCompare(grp.nth(2), [3, 30, 300]);
});

test('Groupby nth uneven', () => {
  const a      = Series.new({type: new Int32, data: [1, 1, 1, 2, 2, 2, 3, 3]});
  const b      = Series.new({type: new Float64, data: [1, 2, 3, 10, 20, 30, 100, 200]});
  const df     = new DataFrame({'a': a, 'b': b});
  const grp    = new GroupBy({obj: df, by: ['a']});
  const result = grp.nth(2)
  basicAggCompare(result, [3, 30, 0]);
  expect(result.get('b').nullCount).toBe(1)
});

test('Groupby nunique basic', () => {
  const df  = makeBasicData([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  const grp = new GroupBy({obj: df, by: ['a']});
  basicAggCompare(grp.nunique(), [3, 4, 3]);
});

test('Groupby std basic', () => {
  const df  = makeBasicData([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  const grp = new GroupBy({obj: df, by: ['a']});
  basicAggCompare(grp.std(), [3, Math.sqrt(131 / 12), Math.sqrt(31 / 3)]);
});

test('Groupby sum basic', () => {
  const df  = makeBasicData([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  const grp = new GroupBy({obj: df, by: ['a']});
  basicAggCompare(grp.sum(), [9, 19, 17]);
});

test('Groupby var basic', () => {
  const df  = makeBasicData([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  const grp = new GroupBy({obj: df, by: ['a']});
  basicAggCompare(grp.var(), [9, 131 / 12, 31 / 3]);
});

export type BasicAggType =
  'sum'|'min'|'max'|'argmin'|'argmax'|'mean'|'count'|'nunique'|'var'|'std'|'median';

const BASIC_AGGS: BasicAggType[] =
  ['sum', 'min', 'max', 'argmin', 'argmax', 'mean', 'count', 'nunique', 'var', 'std', 'median'];

for (const agg of BASIC_AGGS) {
  test(`Groupby ${agg} empty`, () => {
    const a      = Series.new({type: new Int32, data: []});
    const b      = Series.new({type: new Float64, data: []});
    const df     = new DataFrame({'a': a, 'b': b});
    const grp    = new GroupBy({obj: df, by: ['a']});
    const result = grp[agg]();
    expect(result.get('a').length).toBe(0);
    expect(result.get('b').length).toBe(0);
  });
}

test(`Groupby nth empty`, () => {
  const a      = Series.new({type: new Int32, data: []});
  const b      = Series.new({type: new Float64, data: []});
  const df     = new DataFrame({'a': a, 'b': b});
  const grp    = new GroupBy({obj: df, by: ['a']});
  const result = grp.nth(0);
  expect(result.get('a').length).toBe(0);
  expect(result.get('b').length).toBe(0);
});

for (const agg of BASIC_AGGS) {
  test(`Groupby ${agg} null keys`, () => {
    const a      = Series.new({type: new Int32, data: [1, 2, 3], nullMask: [false, false, false]});
    const b      = Series.new({type: new Float64, data: [3, 4, 5]});
    const df     = new DataFrame({'a': a, 'b': b});
    const grp    = new GroupBy({obj: df, by: ['a']});
    const result = grp[agg]();
    expect(result.get('a').length).toBe(0);
    expect(result.get('b').length).toBe(0);
  });
}

test(`Groupby nth null keys`, () => {
  const a      = Series.new({type: new Int32, data: [1, 2, 3], nullMask: [false, false, false]});
  const b      = Series.new({type: new Float64, data: [3, 4, 5]});
  const df     = new DataFrame({'a': a, 'b': b});
  const grp    = new GroupBy({obj: df, by: ['a']});
  const result = grp.nth(0);
  expect(result.get('a').length).toBe(0);
  expect(result.get('b').length).toBe(0);
});

for (const agg of BASIC_AGGS) {
  test(`Groupby ${agg} null values`, () => {
    const a   = Series.new({type: new Int32, data: [1, 1, 1]});
    const b   = Series.new({type: new Float64, data: [3, 4, 5], nullMask: [false, false, false]});
    const df  = new DataFrame({'a': a, 'b': b});
    const grp = new GroupBy({obj: df, by: ['a']});
    const result = grp[agg]();
    expect([...result.get('a').toArrow()]).toEqual([1]);
    expect(result.get('a').nullCount).toBe(0);
    expect(result.get('b').length).toBe(1);
    if (agg == 'count' || agg == 'nunique') {
      expect(result.get('b').nullCount).toBe(0);
    } else {
      expect(result.get('b').nullCount).toBe(1);
    }
  });
}

test(`Groupby nth null values`, () => {
  const a      = Series.new({type: new Int32, data: [1, 1, 1]});
  const b      = Series.new({type: new Float64, data: [3, 4, 5], nullMask: [false, false, false]});
  const df     = new DataFrame({'a': a, 'b': b});
  const grp    = new GroupBy({obj: df, by: ['a']});
  const result = grp.nth(0);
  expect([...result.get('a').toArrow()]).toEqual([1]);
  expect(result.get('a').nullCount).toBe(0);
  expect(result.get('b').length).toBe(1);
  expect(result.get('b').nullCount).toBe(1);
});
