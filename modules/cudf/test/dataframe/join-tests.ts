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
import {DataFrame, Int32, Series} from '@rapidsai/cudf';
import {DeviceBuffer} from '@rapidsai/rmm';

setDefaultAllocator((byteLength: number) => new DeviceBuffer(byteLength));

function makeLeftData() {
  const a = Series.new({type: new Int32, data: [1, 2, 3, 4, 5]});
  const b = Series.new({type: new Int32, data: [0, 0, 1, 1, 2]});
  return new DataFrame({a, b});
}

function makeRightData() {
  const b = Series.new({type: new Int32, data: [0, 1, 3]});
  const c = Series.new({type: new Int32, data: [0, 10, 30]});
  return new DataFrame({b, c});
}

describe('DataFrame.join ', () => {
  test('can inner join', () => {
    const left   = makeLeftData();
    const right  = makeRightData();
    const result = left.join({other: right, on: ['b'], how: 'inner'});
    expect(result.numColumns).toEqual(3);
    expect(result.names).toEqual(['a', 'b', 'c']);
    expect([...result.get('b')]).toEqual([0, 0, 1, 1]);
    expect([...result.get('a')]).toEqual([1, 2, 3, 4]);
    expect([...result.get('c')]).toEqual([0, 0, 10, 10]);
  });

  test('can left join', () => {
    const left   = makeLeftData();
    const right  = makeRightData();
    const result = left.join({other: right, on: ['b'], how: 'left'});
    expect(result.numColumns).toEqual(3);
    expect(result.names).toEqual(['a', 'b', 'c']);
    expect([...result.get('b')]).toEqual([0, 0, 1, 1, 2]);
    expect([...result.get('a')]).toEqual([1, 2, 3, 4, 5]);
    expect([...result.get('c')]).toEqual([0, 0, 10, 10, null]);
  });

  test('can outer join', () => {
    const left   = makeLeftData();
    const right  = makeRightData();
    const result = left.join({other: right, on: ['b'], how: 'outer'});
    expect(result.numColumns).toEqual(3);
    expect(result.names).toEqual(['a', 'b', 'c']);
    expect([...result.get('a')]).toEqual([1, 2, 3, 4, 5, null]);

    // This seems to disagree with pandas and cudf, which has [0, 0, 1, 1, 2, 3]
    expect([...result.get('b')]).toEqual([0, 0, 1, 1, 2, null]);

    expect([...result.get('c')]).toEqual([0, 0, 10, 10, null, 30]);
  });
});
