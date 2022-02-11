// Copyright (c) 2022, NVIDIA CORPORATION.
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

import {setDefaultAllocator} from '@rapidsai/cuda';
import {DataFrame, Series} from '@rapidsai/cudf';
import {DeviceBuffer} from '@rapidsai/rmm';

setDefaultAllocator((byteLength: number) => new DeviceBuffer(byteLength));

describe(`DataFrame.flatten`, () => {
  const input = new DataFrame({
    a: Series.new([100, 200, 300]),
    b: Series.new([[1, 2, 7], [5, 6], [0, 3]]),
    c: Series.new(['string0', 'string1', 'string2']),
    d: Series.new([[1, 2, 7], [5, 6], [0, 3]]),
  });

  test(`doesn't flatten non-list columns`, () => {
    const expected = input.assign({});

    const actual = input.flatten(['a']);

    expect([...actual.get('a')]).toEqual([...expected.get('a')]);
    expect([...actual.get('b')]).toEqual([...expected.get('b')]);
    expect([...actual.get('c')]).toEqual([...expected.get('c')]);
    expect([...actual.get('d')]).toEqual([...expected.get('d')]);
  });

  test(`flattens a single list column`, () => {
    const expected = new DataFrame({
      a: Series.new([100, 100, 100, 200, 200, 300, 300]),
      b: Series.new([1, 2, 7, 5, 6, 0, 3]),
      c: Series.new(['string0', 'string0', 'string0', 'string1', 'string1', 'string2', 'string2']),
      d: Series.new([[1, 2, 7], [1, 2, 7], [1, 2, 7], [5, 6], [5, 6], [0, 3], [0, 3]]),
    });

    const actual = input.flatten(['b']);

    expect([...actual.get('a')]).toEqual([...expected.get('a')]);
    expect([...actual.get('b')]).toEqual([...expected.get('b')]);
    expect([...actual.get('c')]).toEqual([...expected.get('c')]);
    expect([...actual.get('d')]).toEqual([...expected.get('d')]);
  });

  test(`flattens multiple list columns`, () => {
    const expected = new DataFrame({
      a: Series.new(
        [100, 100, 100, 100, 100, 100, 100, 100, 100, 200, 200, 200, 200, 300, 300, 300, 300]),
      b: Series.new([1, 1, 1, 2, 2, 2, 7, 7, 7, 5, 5, 6, 6, 0, 0, 3, 3]),
      c: Series.new([
        'string0',
        'string0',
        'string0',
        'string0',
        'string0',
        'string0',
        'string0',
        'string0',
        'string0',
        'string1',
        'string1',
        'string1',
        'string1',
        'string2',
        'string2',
        'string2',
        'string2'
      ]),
      d: Series.new([1, 2, 7, 1, 2, 7, 1, 2, 7, 5, 6, 5, 6, 0, 3, 0, 3]),
    });

    const actual = input.flatten();

    expect(actual.toString()).toEqual(expected.toString());
  });
});
