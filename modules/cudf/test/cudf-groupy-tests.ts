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

import {setDefaultAllocator, Uint8Buffer} from '@nvidia/cuda';
import {AggFunc, DataFrame, Float64, GroupBy, Int32, Series} from '@nvidia/cudf';
import {CudaMemoryResource, DeviceBuffer} from '@nvidia/rmm';
import {BoolVector} from 'apache-arrow'
import {toBeDeepCloseTo} from 'jest-matcher-deep-close-to';

expect.extend({toBeDeepCloseTo});

const mr = new CudaMemoryResource();

setDefaultAllocator((byteLength: number) => new DeviceBuffer(byteLength, mr));

function makeBasicData(values: number[]): DataFrame {
  const a = Series.new({type: new Int32(), data: new Int32Array([1, 2, 3, 1, 2, 2, 1, 3, 3, 2])});
  const b = Series.new({type: new Float64(), data: new Float64Array(values)});
  return new DataFrame({"a": a, "b": b});
}

function basicAggCompare(result: DataFrame, expected: number[]): void {
  const ra = result.get("a");
  const rb = result.get("b");

  const a_expected = Series.new({type: new Int32(), data: new Int32Array([1, 2, 3])});
  expect([...ra.toArrow()]).toEqual([...a_expected.toArrow()]);

  const b_expected = Series.new({type: new Float64(), data: new Float64Array(expected)});
  expect([...rb.toArrow()]).toBeDeepCloseTo([...b_expected.toArrow()], 12);
}

test('Groupby basic', () => {
  const a  = Series.new({type: new Int32(), data: new Int32Array([1, 1, 2, 1, 2, 3])});
  const df = new DataFrame({"a": a});

  const grp = new GroupBy({obj: df, by: ["a"]});

  const groups = grp.getGroups();

  const keys_result = Series.new(groups["keys"].get("a"));
  expect([...keys_result.toArrow()]).toEqual([1, 1, 1, 2, 2, 3]);

  expect(groups.values).toBeUndefined();

  expect([...groups["offsets"]]).toEqual([0, 3, 5, 6]);
});

test('Groupby empty', () => {
  const a  = Series.new({type: new Int32(), data: new Int32Array([])});
  const df = new DataFrame({"a": a});

  const grp = new GroupBy({obj: df, by: ["a"]});

  const groups = grp.getGroups();

  const keys_result = Series.new(groups["keys"].get("a"));
  expect(keys_result.length).toBe(0);

  expect(groups.values).toBeUndefined();

  expect([...groups["offsets"]]).toEqual([0]);
});

test('Groupby basic with values', () => {
  const a  = Series.new({type: new Int32(), data: new Int32Array([5, 4, 3, 2, 1, 0])});
  const b  = Series.new({type: new Int32(), data: new Int32Array([0, 0, 1, 1, 2, 2])});
  const df = new DataFrame({"a": a, "b": b});

  const grp = new GroupBy({obj: df, by: ["a"]});

  const groups = grp.getGroups();

  const keys_result = Series.new(groups["keys"].get("a"));
  expect([...keys_result.toArrow()]).toEqual([0, 1, 2, 3, 4, 5]);

  const values_result = Series.new(groups.values?.get("b"));
  expect([...values_result.toArrow()]).toEqual([2, 2, 1, 1, 0, 0]);

  expect([...groups["offsets"]]).toEqual([0, 1, 2, 3, 4, 5, 6]);
});

test('Groupby all nulls', () => {
  const a  = Series.new({
    type: new Int32(),
    data: new Int32Array([1, 1, 2, 3, 1, 2]),
    nullMask: new Uint8Buffer(BoolVector.from([0, 0, 0, 0, 0, 0]).values),
  });
  const df = new DataFrame({"a": a});

  const grp = new GroupBy({obj: df, by: ["a"]});

  const groups = grp.getGroups();

  const keys_result = Series.new(groups["keys"].get("a"));
  expect(keys_result.length).toBe(0);

  expect(groups.values).toBeUndefined();

  expect([...groups["offsets"]]).toEqual([0]);
});

test('Groupby some nulls', () => {
  const a  = Series.new({
    type: new Int32(),
    data: new Int32Array([1, 1, 3, 2, 1, 2]),
    nullMask: new Uint8Buffer(BoolVector.from([1, 0, 1, 0, 0, 1]).values),
  });
  const b  = Series.new({type: new Int32(), data: new Int32Array([1, 2, 3, 4, 5, 6])});
  const df = new DataFrame({"a": a, "b": b});

  const grp = new GroupBy({obj: df, by: ["a"]});

  const groups = grp.getGroups();

  const keys_result = Series.new(groups["keys"].get("a"));
  expect([...keys_result.toArrow()]).toEqual([1, 2, 3]);
  expect(keys_result.nullCount).toBe(0)

  const values_result = Series.new(groups.values?.get("b"));
  expect([...values_result.toArrow()]).toEqual([1, 6, 3]);

  expect([...groups["offsets"]]).toEqual([0, 1, 2, 3]);
});

test('Groupby argmax basic', () => {
  const df  = makeBasicData([9, 8, 7, 6, 5, 4, 3, 2, 1, 0]);
  const grp = new GroupBy({obj: df, by: ["a"]});
  basicAggCompare(grp.argmax(), [0, 1, 2]);
});

test('Groupby argmin basic', () => {
  const df  = makeBasicData([9, 8, 7, 6, 5, 4, 3, 2, 1, 0]);
  const grp = new GroupBy({obj: df, by: ["a"]});
  basicAggCompare(grp.argmin(), [6, 9, 8]);
});

test('Groupby count basic', () => {
  const df  = makeBasicData([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  const grp = new GroupBy({obj: df, by: ["a"]});
  basicAggCompare(grp.count(), [3, 4, 3]);
});

test('Groupby max basic', () => {
  const df  = makeBasicData([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  const grp = new GroupBy({obj: df, by: ["a"]});
  basicAggCompare(grp.max(), [6, 9, 8]);
});

test('Groupby mean basic', () => {
  const df  = makeBasicData([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  const grp = new GroupBy({obj: df, by: ["a"]});
  basicAggCompare(grp.mean(), [3, 19 / 4, 17 / 3]);
});

test('Groupby median basic', () => {
  const df  = makeBasicData([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  const grp = new GroupBy({obj: df, by: ["a"]});
  basicAggCompare(grp.median(), [3, 4.5, 7]);
});

test('Groupby min basic', () => {
  const df  = makeBasicData([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  const grp = new GroupBy({obj: df, by: ["a"]});
  basicAggCompare(grp.min(), [0, 1, 2]);
});

test('Groupby nunique basic', () => {
  const df  = makeBasicData([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  const grp = new GroupBy({obj: df, by: ["a"]});
  basicAggCompare(grp.nunique(), [3, 4, 3]);
});

test('Groupby std basic', () => {
  const df  = makeBasicData([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  const grp = new GroupBy({obj: df, by: ["a"]});
  basicAggCompare(grp.std(), [3, Math.sqrt(131 / 12), Math.sqrt(31 / 3)]);
});

test('Groupby sum basic', () => {
  const df  = makeBasicData([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  const grp = new GroupBy({obj: df, by: ["a"]});
  basicAggCompare(grp.sum(), [9, 19, 17]);
});

test('Groupby var basic', () => {
  const df  = makeBasicData([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  const grp = new GroupBy({obj: df, by: ["a"]});
  basicAggCompare(grp.var(), [9, 131 / 12, 31 / 3]);
});

const BASIC_AGGS: AggFunc[] =
  ["sum", "min", "max", "argmin", "argmax", "mean", "count", "nunique", "var", "std", "median"];

for (const agg of BASIC_AGGS) {
  test(`Groupby ${agg} empty`, () => {
    const a      = Series.new({type: new Int32(), data: new Int32Array([])});
    const b      = Series.new({type: new Float64(), data: new Float64Array([])});
    const df     = new DataFrame({"a": a, "b": b});
    const grp    = new GroupBy({obj: df, by: ["a"]});
    const result = grp[agg]();
    expect(result.get("a").length).toBe(0);
    expect(result.get("b").length).toBe(0);
  });
}

for (const agg of BASIC_AGGS) {
  test(`Groupby ${agg} null keys`, () => {
    const mask   = new Uint8Buffer(BoolVector.from([0, 0, 0]).values);
    const a      = Series.new({type: new Int32(), data: new Int32Array([1, 2, 3]), nullMask: mask});
    const b      = Series.new({type: new Float64(), data: new Float64Array([3, 4, 5])});
    const df     = new DataFrame({"a": a, "b": b});
    const grp    = new GroupBy({obj: df, by: ["a"]});
    const result = grp[agg]();
    expect(result.get("a").length).toBe(0);
    expect(result.get("b").length).toBe(0);
  });
}

for (const agg of BASIC_AGGS) {
  test(`Groupby ${agg} null values`, () => {
    const mask = new Uint8Buffer(BoolVector.from([0, 0, 0]).values);
    const a    = Series.new({type: new Int32(), data: new Int32Array([1, 1, 1])});
    const b  = Series.new({type: new Float64(), data: new Float64Array([3, 4, 5]), nullMask: mask});
    const df = new DataFrame({"a": a, "b": b});
    const grp    = new GroupBy({obj: df, by: ["a"]});
    const result = grp[agg]();
    expect([...result.get("a").toArrow()]).toEqual([1]);
    expect(result.get("a").nullCount).toBe(0);
    expect(result.get("b").length).toBe(1);
    if (agg == "count" || agg == "nunique") {
      expect(result.get("b").nullCount).toBe(0);
    } else {
      expect(result.get("b").nullCount).toBe(1);
    }
  });
}
