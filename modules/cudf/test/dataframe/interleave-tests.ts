// Copyright (c) 2021, NVIDIA CORPORATION.
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

import {setDefaultAllocator} from '@nvidia/cuda';
import {DataFrame, Float64, Int32, List, Series} from '@rapidsai/cudf';
import {DeviceBuffer} from '@rapidsai/rmm';

setDefaultAllocator((byteLength: number) => new DeviceBuffer(byteLength));

describe('DataFrame.interleaveColumns', () => {
  test(`interleaves Float64 columns`, () => {
    const df       = new DataFrame({
      a: Series.new([1, 2, 3]),
      b: Series.new([4, 5, 6]),
    });
    const expected = [1, 4, 2, 5, 3, 6];
    const actual   = df.interleaveColumns();
    expect([...actual]).toEqual(expected);
    expectFloat64(actual.type);
    // Cause a compile error if `df.interleaveColumns()` type handling fails, e.g.
    // ```ts
    // expectFloat64(<Int32|Float64>new Float64());
    // ```
    function expectFloat64(type: Float64) { expect(type).toBeInstanceOf(Float64); }
  });

  test(`interleaves List columns`, () => {
    const df       = new DataFrame({
      a: Series.new([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
      b: Series.new([[10, 11, 12], [13, 14, 15], [16, 17, 18]]),
    });
    const expected = [
      [0, 1, 2],
      [10, 11, 12],
      [3, 4, 5],
      [13, 14, 15],
      [6, 7, 8],
      [16, 17, 18],
    ];
    const actual = df.interleaveColumns();
    expect([...actual].map((x) => x ? [...x] : null)).toEqual(expected);
    expectListOfFloat64(actual.type);
    // Cause a compile error if `df.interleaveColumns()` type handling fails, e.g.
    // ```ts
    // expectListOfFloat64(<Int32|Float64>new List(new Field('', new Int32)));
    // ```
    function expectListOfFloat64(type: List<Float64>) {
      expect(type).toBeInstanceOf(List);
      expect(type.children[0].type).toBeInstanceOf(Float64);
    }
  });

  test(`casts mixed types to the input dtype`, () => {
    const df       = new DataFrame({
      a: Series.new([1, 2, 3]).cast(new Int32),
      b: Series.new([4, 5, 6]),
    });
    const expected = [1, 4, 2, 5, 3, 6];
    const actual   = df.interleaveColumns(df.types['a']);
    expect([...actual]).toEqual(expected);
    expectInt32(actual.type);
    // Cause a compile error if `df.interleaveColumns()` type handling fails, e.g.
    // ```ts
    // expectInt32(<Int32|Float64>new Int32());
    // ```
    function expectInt32(type: Int32) { expect(type).toBeInstanceOf(Int32); }
  });

  test(`throws an error if mixed types and no input dtype`, () => {
    const df = new DataFrame({
      a: Series.new([1, 2, 3]).cast(new Int32),
      b: Series.new([4, 5, 6]),
    });
    expect(() => df.interleaveColumns()).toThrow();
  });
});
