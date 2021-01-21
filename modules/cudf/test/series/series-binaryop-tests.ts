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
import {Int32, Series} from '@nvidia/cudf';
import {DeviceBuffer} from '@nvidia/rmm';
import {IntVector} from 'apache-arrow';

setDefaultAllocator((byteLength: number) => new DeviceBuffer(byteLength));

const makeTestSeries = (values: number[]) => {
  const src = IntVector.from(new Int32Array(values));
  const lhs = new Series({type: new Int32, data: new DeviceBuffer(src.values)});
  const rhs = new Series({type: new Int32, data: new DeviceBuffer(src.values.map((x => x + 1)))});
  return {lhs, rhs};
};

describe('Series.eq', () => {
  test('compares against Series', () => {
    const {lhs, rhs} = makeTestSeries([0, 1, 2]);
    // lhs == lhs == true
    expect([...lhs.eq(lhs).toArrow()]).toEqual([true, true, true]);
    // lhs == rhs == false
    expect([...lhs.eq(rhs).toArrow()]).toEqual([false, false, false]);
  });
  test('compares against scalars', () => {
    const {lhs} = makeTestSeries([0, 1, 2]);
    expect([...lhs.eq(0).toArrow()]).toEqual([true, false, false]);
    expect([...lhs.eq(1).toArrow()]).toEqual([false, true, false]);
    expect([...lhs.eq(2).toArrow()]).toEqual([false, false, true]);
  });
});

describe('Series.lt', () => {
  test('compares against Series', () => {
    const {lhs, rhs} = makeTestSeries([0, 1, 2]);
    // lhs < rhs == true
    expect([...lhs.lt(rhs).toArrow()]).toEqual([true, true, true]);
    // lhs < lhs == false
    expect([...lhs.lt(lhs).toArrow()]).toEqual([false, false, false]);
    // rhs < lhs == false
    expect([...rhs.lt(rhs).toArrow()]).toEqual([false, false, false]);
  });
  test('compares against scalars', () => {
    const {lhs} = makeTestSeries([0, 1, 2]);
    expect([...lhs.lt(3).toArrow()]).toEqual([true, true, true]);
    expect([...lhs.lt(2).toArrow()]).toEqual([true, true, false]);
    expect([...lhs.lt(1).toArrow()]).toEqual([true, false, false]);
    expect([...lhs.lt(0).toArrow()]).toEqual([false, false, false]);
  });
});

describe('Series.le', () => {
  test('compares against Series', () => {
    const {lhs, rhs} = makeTestSeries([0, 1, 2]);
    // lhs <= lhs == true
    expect([...lhs.le(lhs).toArrow()]).toEqual([true, true, true]);
    // lhs <= rhs == true
    expect([...lhs.le(rhs).toArrow()]).toEqual([true, true, true]);
    // rhs <= lhs == false
    expect([...rhs.le(lhs).toArrow()]).toEqual([false, false, false]);
  });
  test('compares against scalars', () => {
    const {lhs} = makeTestSeries([0, 1, 2]);
    expect([...lhs.le(2).toArrow()]).toEqual([true, true, true]);
    expect([...lhs.le(1).toArrow()]).toEqual([true, true, false]);
    expect([...lhs.le(0).toArrow()]).toEqual([true, false, false]);
  });
});

describe('Series.gt', () => {
  test('compares against Series', () => {
    const {lhs, rhs} = makeTestSeries([0, 1, 2]);
    // rhs > lhs == true
    expect([...rhs.gt(lhs).toArrow()]).toEqual([true, true, true]);
    // lhs > rhs == false
    expect([...lhs.gt(rhs).toArrow()]).toEqual([false, false, false]);
    // lhs > lhs == false
    expect([...lhs.gt(lhs).toArrow()]).toEqual([false, false, false]);
  });
  test('compares against scalars', () => {
    const {lhs} = makeTestSeries([0, 1, 2]);
    expect([...lhs.gt(2).toArrow()]).toEqual([false, false, false]);
    expect([...lhs.gt(1).toArrow()]).toEqual([false, false, true]);
    expect([...lhs.gt(0).toArrow()]).toEqual([false, true, true]);
    expect([...lhs.gt(-1).toArrow()]).toEqual([true, true, true]);
  });
});

describe('Series.ge', () => {
  test('compares against Series', () => {
    const {lhs, rhs} = makeTestSeries([0, 1, 2]);
    // lhs >= lhs == true
    expect([...lhs.ge(lhs).toArrow()]).toEqual([true, true, true]);
    // lhs >= rhs == false
    expect([...lhs.ge(rhs).toArrow()]).toEqual([false, false, false]);
  });
  test('compares against scalars', () => {
    const {lhs} = makeTestSeries([0, 1, 2]);
    expect([...lhs.ge(3).toArrow()]).toEqual([false, false, false]);
    expect([...lhs.ge(2).toArrow()]).toEqual([false, false, true]);
    expect([...lhs.ge(1).toArrow()]).toEqual([false, true, true]);
    expect([...lhs.ge(0).toArrow()]).toEqual([true, true, true]);
  });
});
