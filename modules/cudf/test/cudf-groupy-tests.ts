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

import {setDefaultAllocator, Uint8Buffer} from '@nvidia/cuda';
import {DataFrame, GroupBy, Int32, Series} from '@nvidia/cudf';
import {CudaMemoryResource, DeviceBuffer} from '@nvidia/rmm';
import {BoolVector} from 'apache-arrow'

const mr = new CudaMemoryResource();

setDefaultAllocator((byteLength: number) => new DeviceBuffer(byteLength, mr));

test('Groupby basic', () => {
  const a  = Series.new({type: new Int32(), data: new Int32Array([1, 1, 2, 1, 2, 3])});
  const df = new DataFrame({"a": a});

  const grp = new GroupBy({keys: df});

  const groups = grp.getGroups();

  const keys_result = Series.new(groups["keys"].getColumnByIndex(0));
  expect([...keys_result.toArrow()]).toEqual([1, 1, 1, 2, 2, 3]);

  expect(groups["values"]).toBeUndefined();

  expect([...groups["offsets"]]).toEqual([0, 3, 5, 6]);
});

test('Groupby empty', () => {
  const a  = Series.new({type: new Int32(), data: new Int32Array([])});
  const df = new DataFrame({"a": a});

  const grp = new GroupBy({keys: df});

  const groups = grp.getGroups();

  const keys_result = Series.new(groups["keys"].getColumnByIndex(0));
  expect(keys_result.length).toBe(0);

  expect(groups["values"]).toBeUndefined();

  expect([...groups["offsets"]]).toEqual([0]);
});

test('Groupby basic with values', () => {
  const a  = Series.new({type: new Int32(), data: new Int32Array([5, 4, 3, 2, 1, 0])});
  const df = new DataFrame({"a": a});

  const b   = Series.new({type: new Int32(), data: new Int32Array([0, 0, 1, 1, 2, 2])});
  const df2 = new DataFrame({"b": b});

  const grp = new GroupBy({keys: df});

  const groups = grp.getGroups(df2);

  const keys_result = Series.new(groups["keys"].getColumnByIndex(0));
  expect([...keys_result.toArrow()]).toEqual([0, 1, 2, 3, 4, 5]);

  const values_result = Series.new(groups["values"].getColumnByIndex(0));
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

  const grp = new GroupBy({keys: df});

  const groups = grp.getGroups();

  const keys_result = Series.new(groups["keys"].getColumnByIndex(0));
  expect(keys_result.length).toBe(0);

  expect(groups["values"]).toBeUndefined();

  expect([...groups["offsets"]]).toEqual([0]);
});

test('Groupby some nulls', () => {
  const a  = Series.new({
    type: new Int32(),
    data: new Int32Array([1, 1, 3, 2, 1, 2]),
    nullMask: new Uint8Buffer(BoolVector.from([1, 0, 1, 0, 0, 1]).values),
  });
  const df = new DataFrame({"a": a});

  const b   = Series.new({type: new Int32(), data: new Int32Array([1, 2, 3, 4, 5, 6])});
  const df2 = new DataFrame({"b": b});

  const grp = new GroupBy({keys: df});

  const groups = grp.getGroups(df2);

  const keys_result = Series.new(groups["keys"].getColumnByIndex(0));
  expect([...keys_result.toArrow()]).toEqual([1, 2, 3]);
  expect(keys_result.nullCount).toBe(0)

  const values_result = Series.new(groups["values"].getColumnByIndex(0));
  expect([...values_result.toArrow()]).toEqual([1, 6, 3]);

  expect([...groups["offsets"]]).toEqual([0, 1, 2, 3]);
});
