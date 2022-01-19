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

import {setDefaultAllocator} from '@rapidsai/cuda';
import {DataFrame, Float32, Float64, Int32, Series} from '@rapidsai/cudf';
import {DeviceBuffer} from '@rapidsai/rmm';

setDefaultAllocator((byteLength: number) => new DeviceBuffer(byteLength));

const left = new DataFrame({
  a: Series.new({type: new Int32, data: [1, 2, 3, 4, 5]}),
  b: Series.new({type: new Int32, data: [0, 0, 1, 1, 2]})
});

const right = new DataFrame({
  b: Series.new({type: new Int32, data: [0, 1, 3]}),
  c: Series.new({type: new Int32, data: [0, 10, 30]})
});

const right_conflict = new DataFrame({
  b: Series.new({type: new Int32, data: [0, 1, 3]}),
  a: Series.new({type: new Int32, data: [0, 10, 30]})
});

const left_double = new DataFrame({
  a: Series.new({type: new Int32, data: [0, 1, 1]}),
  b: Series.new({type: new Int32, data: [10, 20, 10]}),
  c: Series.new({type: new Float64, data: [1., 2., 3.]})
});

const right_double = new DataFrame({
  a: Series.new({type: new Int32, data: [0, 0, 1]}),
  b: Series.new({type: new Int32, data: [10, 20, 20]}),
  d: Series.new({type: new Float64, data: [10, 20, 30]})
});

const right_double_conflict = new DataFrame({
  a: Series.new({type: new Int32, data: [0, 0, 1]}),
  b: Series.new({type: new Int32, data: [10, 20, 20]}),
  c: Series.new({type: new Float64, data: [10, 20, 30]})
});

describe('DataFrame.join({how="inner"}) ', () => {
  test('can join with no column name conflicts', () => {
    const result = left.join({other: right, on: ['b'], how: 'inner'});
    expect(result.numColumns).toEqual(3);
    expect(result.names).toEqual(expect.arrayContaining(['a', 'b', 'c']));
    expect([...result.get('b')]).toEqual([0, 0, 1, 1]);
    expect([...result.get('a')]).toEqual([1, 2, 3, 4]);
    expect([...result.get('c')]).toEqual([0, 0, 10, 10]);
  });

  test('discards right conflicts without suffices', () => {
    const result = left.join({other: right_conflict, on: ['b'], how: 'inner'});
    expect(result.numColumns).toEqual(2);
    expect(result.names).toEqual(expect.arrayContaining(['a', 'b']));
    expect([...result.get('b')]).toEqual([0, 0, 1, 1]);
    expect([...result.get('a')]).toEqual([1, 2, 3, 4]);
  });

  test('applies lsuffix', () => {
    const result = left.join({other: right_conflict, on: ['b'], how: 'inner', lsuffix: '_L'});
    expect(result.numColumns).toEqual(3);
    expect(result.names).toEqual(expect.arrayContaining(['a', 'b', 'a_L']));
    expect([...result.get('b')]).toEqual([0, 0, 1, 1]);
    expect([...result.get('a_L')]).toEqual([1, 2, 3, 4]);
    expect([...result.get('a')]).toEqual([0, 0, 10, 10]);
  });

  test('applies rsuffix', () => {
    const result = left.join({other: right_conflict, on: ['b'], how: 'inner', rsuffix: '_R'});
    expect(result.numColumns).toEqual(3);
    expect(result.names).toEqual(expect.arrayContaining(['a', 'b', 'a_R']));
    expect([...result.get('b')]).toEqual([0, 0, 1, 1]);
    expect([...result.get('a')]).toEqual([1, 2, 3, 4]);
    expect([...result.get('a_R')]).toEqual([0, 0, 10, 10]);
  });

  test('applies lsuffix and rsuffix', () => {
    const result =
      left.join({other: right_conflict, on: ['b'], how: 'inner', lsuffix: '_L', rsuffix: '_R'});
    expect(result.numColumns).toEqual(3);
    expect(result.names).toEqual(expect.arrayContaining(['a_L', 'b', 'a_R']));
    expect([...result.get('b')]).toEqual([0, 0, 1, 1]);
    expect([...result.get('a_L')]).toEqual([1, 2, 3, 4]);
    expect([...result.get('a_R')]).toEqual([0, 0, 10, 10]);
  });

  test('can join on multi-index', () => {
    const result = left_double.join({other: right_double, on: ['a', 'b'], how: 'inner'});
    expect(result.numColumns).toEqual(4);
    expect(result.names).toEqual(expect.arrayContaining(['a', 'b', 'c', 'd']));
    expect([...result.get('a')]).toEqual([0, 1]);
    expect([...result.get('b')]).toEqual([10, 20]);
    expect([...result.get('c')]).toEqual([1, 2]);
    expect([...result.get('d')]).toEqual([10, 30]);
  });

  test('can join on multi-index with conflict, rsuffix', () => {
    const result =
      left_double.join({other: right_double_conflict, on: ['a', 'b'], how: 'inner', rsuffix: '_R'});
    expect(result.numColumns).toEqual(4);
    expect(result.names).toEqual(expect.arrayContaining(['a', 'b', 'c', 'c_R']));
    expect([...result.get('a')]).toEqual([0, 1]);
    expect([...result.get('b')]).toEqual([10, 20]);
    expect([...result.get('c')]).toEqual([1, 2]);
    expect([...result.get('c_R')]).toEqual([10, 30]);
  });

  test('can join on multi-index with conflict, lsuffix', () => {
    const result =
      left_double.join({other: right_double_conflict, on: ['a', 'b'], how: 'inner', lsuffix: '_L'});
    expect(result.numColumns).toEqual(4);
    expect(result.names).toEqual(expect.arrayContaining(['a', 'b', 'c_L', 'c']));
    expect([...result.get('a')]).toEqual([0, 1]);
    expect([...result.get('b')]).toEqual([10, 20]);
    expect([...result.get('c_L')]).toEqual([1, 2]);
    expect([...result.get('c')]).toEqual([10, 30]);
  });

  test('can join on multi-index with conflict, lsuffix and rsuffix', () => {
    const result = left_double.join(
      {other: right_double_conflict, on: ['a', 'b'], how: 'inner', lsuffix: '_L', rsuffix: '_R'});
    expect(result.numColumns).toEqual(4);
    expect(result.names).toEqual(expect.arrayContaining(['a', 'b', 'c_L', 'c_R']));
    expect([...result.get('a')]).toEqual([0, 1]);
    expect([...result.get('b')]).toEqual([10, 20]);
    expect([...result.get('c_L')]).toEqual([1, 2]);
    expect([...result.get('c_R')]).toEqual([10, 30]);
  });

  test('can find common type for single-index join', () => {
    const left_float =
      new DataFrame({a: Series.new([1, 2, 3, 4, 5]), b: Series.new([0, 0, 1, 1, 2])});

    const result = left_float.join({other: right, on: ['b'], how: 'inner'});
    expect(result.numColumns).toEqual(3);
    expect(result.names).toEqual(expect.arrayContaining(['a', 'b', 'c']));
    expect([...result.get('b')]).toEqual([0, 0, 1, 1]);
    expect([...result.get('a')]).toEqual([1, 2, 3, 4]);
    expect([...result.get('c')]).toEqual([0, 0, 10, 10]);
  });

  test('can find common type on multi-index join', () => {
    const left_double_float = new DataFrame({
      a: Series.new([0, 1, 1]),
      b: Series.new([10, 20, 10]),
      c: Series.new({type: new Float32, data: [1., 2., 3.]})
    });

    const result = left_double_float.join({other: right_double, on: ['a', 'b'], how: 'inner'});
    expect(result.numColumns).toEqual(4);
    expect(result.names).toEqual(expect.arrayContaining(['a', 'b', 'c', 'd']));
    expect([...result.get('a')]).toEqual([0, 1]);
    expect([...result.get('b')]).toEqual([10, 20]);
    expect([...result.get('c')]).toEqual([1, 2]);
    expect([...result.get('d')]).toEqual([10, 30]);
  });
});

describe('DataFrame.join({how="left"}) ', () => {
  test('can join with no column name conflicts', () => {
    const result = left.join({other: right, on: ['b'], how: 'left'});
    expect(result.numColumns).toEqual(3);
    expect(result.names).toEqual(expect.arrayContaining(['a', 'b', 'c']));
    expect([...result.get('b')]).toEqual([0, 0, 1, 1, 2]);
    expect([...result.get('a')]).toEqual([1, 2, 3, 4, 5]);
    expect([...result.get('c')]).toEqual([0, 0, 10, 10, null]);
  });

  test('discards right conflicts without suffices', () => {
    const result = left.join({other: right_conflict, on: ['b'], how: 'left'});
    expect(result.numColumns).toEqual(2);
    expect(result.names).toEqual(expect.arrayContaining(['a', 'b']));
    expect([...result.get('b')]).toEqual([0, 0, 1, 1, 2]);
    expect([...result.get('a')]).toEqual([1, 2, 3, 4, 5]);
  });

  test('applies lsuffix', () => {
    const result = left.join({other: right_conflict, on: ['b'], how: 'left', lsuffix: '_L'});
    expect(result.numColumns).toEqual(3);
    expect(result.names).toEqual(expect.arrayContaining(['a', 'b', 'a_L']));
    expect([...result.get('b')]).toEqual([0, 0, 1, 1, 2]);
    expect([...result.get('a_L')]).toEqual([1, 2, 3, 4, 5]);
    expect([...result.get('a')]).toEqual([0, 0, 10, 10, null]);
  });

  test('applies rsuffix', () => {
    const result = left.join({other: right_conflict, on: ['b'], how: 'left', rsuffix: '_R'});
    expect(result.numColumns).toEqual(3);
    expect(result.names).toEqual(expect.arrayContaining(['a', 'b', 'a_R']));
    expect([...result.get('b')]).toEqual([0, 0, 1, 1, 2]);
    expect([...result.get('a')]).toEqual([1, 2, 3, 4, 5]);
    expect([...result.get('a_R')]).toEqual([0, 0, 10, 10, null]);
  });

  test('applies lsuffix and rsuffix', () => {
    const result =
      left.join({other: right_conflict, on: ['b'], how: 'left', lsuffix: '_L', rsuffix: '_R'});
    expect(result.numColumns).toEqual(3);
    expect(result.names).toEqual(expect.arrayContaining(['a_L', 'b', 'a_R']));
    expect([...result.get('b')]).toEqual([0, 0, 1, 1, 2]);
    expect([...result.get('a_L')]).toEqual([1, 2, 3, 4, 5]);
    expect([...result.get('a_R')]).toEqual([0, 0, 10, 10, null]);
  });

  test('can join on multi-index', () => {
    const result = left_double.join({other: right_double, on: ['a', 'b'], how: 'left'});
    expect(result.numColumns).toEqual(4);
    expect(result.names).toEqual(expect.arrayContaining(['a', 'b', 'c', 'd']));
    expect([...result.get('a')]).toEqual([0, 1, 1]);
    expect([...result.get('b')]).toEqual([10, 20, 10]);
    expect([...result.get('c')]).toEqual([1, 2, 3]);
    expect([...result.get('d')]).toEqual([10, 30, null]);
  });

  test('can join on multi-index with conflict, rsuffix', () => {
    const result =
      left_double.join({other: right_double_conflict, on: ['a', 'b'], how: 'left', rsuffix: '_R'});
    expect(result.numColumns).toEqual(4);
    expect(result.names).toEqual(expect.arrayContaining(['a', 'b', 'c', 'c_R']));
    expect([...result.get('a')]).toEqual([0, 1, 1]);
    expect([...result.get('b')]).toEqual([10, 20, 10]);
    expect([...result.get('c')]).toEqual([1, 2, 3]);
    expect([...result.get('c_R')]).toEqual([10, 30, null]);
  });

  test('can join on multi-index with conflict, lsuffix', () => {
    const result =
      left_double.join({other: right_double_conflict, on: ['a', 'b'], how: 'left', lsuffix: '_L'});
    expect(result.numColumns).toEqual(4);
    expect(result.names).toEqual(expect.arrayContaining(['a', 'b', 'c_L', 'c']));
    expect([...result.get('a')]).toEqual([0, 1, 1]);
    expect([...result.get('b')]).toEqual([10, 20, 10]);
    expect([...result.get('c_L')]).toEqual([1, 2, 3]);
    expect([...result.get('c')]).toEqual([10, 30, null]);
  });

  test('can join on multi-index with conflict, lsuffix and rsuffix', () => {
    const result = left_double.join(
      {other: right_double_conflict, on: ['a', 'b'], how: 'left', lsuffix: '_L', rsuffix: '_R'});
    expect(result.numColumns).toEqual(4);
    expect(result.names).toEqual(expect.arrayContaining(['a', 'b', 'c_L', 'c_R']));
    expect([...result.get('a')]).toEqual([0, 1, 1]);
    expect([...result.get('b')]).toEqual([10, 20, 10]);
    expect([...result.get('c_L')]).toEqual([1, 2, 3]);
    expect([...result.get('c_R')]).toEqual([10, 30, null]);
  });

  test('can find common type for single-index join', () => {
    const left_double =
      new DataFrame({a: Series.new([1, 2, 3, 4, 5]), b: Series.new([0, 0, 1, 1, 2])});

    const result = left_double.join({other: right, on: ['b'], how: 'left'});
    expect(result.numColumns).toEqual(3);
    expect(result.names).toEqual(expect.arrayContaining(['a', 'b', 'c']));
    expect([...result.get('b')]).toEqual([0, 0, 1, 1, 2]);
    expect([...result.get('a')]).toEqual([1, 2, 3, 4, 5]);
    expect([...result.get('c')]).toEqual([0, 0, 10, 10, null]);
  });

  test('can find common type on multi-index join', () => {
    const left_double_float = new DataFrame({
      a: Series.new([0, 1, 1]),
      b: Series.new([10, 20, 10]),
      c: Series.new({type: new Float32, data: [1., 2., 3.]})
    });

    const result = left_double_float.join({other: right_double, on: ['a', 'b'], how: 'left'});
    expect(result.numColumns).toEqual(4);
    expect(result.names).toEqual(expect.arrayContaining(['a', 'b', 'c', 'd']));
    expect([...result.get('a')]).toEqual([0, 1, 1]);
    expect([...result.get('b')]).toEqual([10, 20, 10]);
    expect([...result.get('c')]).toEqual([1, 2, 3]);
    expect([...result.get('d')]).toEqual([10, 30, null]);
  });
});

describe('DataFrame.join({how="outer"}) ', () => {
  test('can join with no column name conflicts', () => {
    const result = left.join({other: right, on: ['b'], how: 'outer'});
    expect(result.numColumns).toEqual(3);
    expect(result.names).toEqual(expect.arrayContaining(['a', 'b', 'c']));
    expect([...result.get('b')]).toEqual([0, 0, 1, 1, 2, 3]);
    expect([...result.get('a')]).toEqual([1, 2, 3, 4, 5, null]);
    expect([...result.get('c')]).toEqual([0, 0, 10, 10, null, 30]);
  });
  test('discards right conflicts without suffices', () => {
    const result = left.join({other: right_conflict, on: ['b'], how: 'outer'});
    expect(result.numColumns).toEqual(2);
    expect(result.names).toEqual(expect.arrayContaining(['a', 'b']));
    expect([...result.get('b')]).toEqual([0, 0, 1, 1, 2, 3]);
    expect([...result.get('a')]).toEqual([1, 2, 3, 4, 5, null]);
  });

  test('applies lsuffix', () => {
    const result = left.join({other: right_conflict, on: ['b'], how: 'outer', lsuffix: '_L'});
    expect(result.numColumns).toEqual(3);
    expect(result.names).toEqual(expect.arrayContaining(['a', 'b', 'a_L']));
    expect([...result.get('b')]).toEqual([0, 0, 1, 1, 2, 3]);
    expect([...result.get('a_L')]).toEqual([1, 2, 3, 4, 5, null]);
    expect([...result.get('a')]).toEqual([0, 0, 10, 10, null, 30]);
  });

  test('applies rsuffix', () => {
    const result = left.join({other: right_conflict, on: ['b'], how: 'outer', rsuffix: '_R'});
    expect(result.numColumns).toEqual(3);
    expect(result.names).toEqual(expect.arrayContaining(['a', 'b', 'a_R']));
    expect([...result.get('b')]).toEqual([0, 0, 1, 1, 2, 3]);
    expect([...result.get('a')]).toEqual([1, 2, 3, 4, 5, null]);
    expect([...result.get('a_R')]).toEqual([0, 0, 10, 10, null, 30]);
  });

  test('applies lsuffix and rsuffix', () => {
    const result =
      left.join({other: right_conflict, on: ['b'], how: 'outer', lsuffix: '_L', rsuffix: '_R'});
    expect(result.numColumns).toEqual(3);
    expect(result.names).toEqual(expect.arrayContaining(['a_L', 'b', 'a_R']));
    expect([...result.get('b')]).toEqual([0, 0, 1, 1, 2, 3]);
    expect([...result.get('a_L')]).toEqual([1, 2, 3, 4, 5, null]);
    expect([...result.get('a_R')]).toEqual([0, 0, 10, 10, null, 30]);
  });

  test('can join on multi-index', () => {
    const result = left_double.join({other: right_double, on: ['a', 'b'], how: 'outer'});
    expect(result.numColumns).toEqual(4);
    expect(result.names).toEqual(expect.arrayContaining(['a', 'b', 'c', 'd']));
    expect([...result.get('a')]).toEqual([0, 1, 1, 0]);
    expect([...result.get('b')]).toEqual([10, 20, 10, 20]);
    expect([...result.get('c')]).toEqual([1, 2, 3, null]);
    expect([...result.get('d')]).toEqual([10, 30, null, 20]);
  });

  test('can join on multi-index with conflict, rsuffix', () => {
    const result =
      left_double.join({other: right_double_conflict, on: ['a', 'b'], how: 'outer', rsuffix: '_R'});
    expect(result.numColumns).toEqual(4);
    expect(result.names).toEqual(expect.arrayContaining(['a', 'b', 'c', 'c_R']));
    expect([...result.get('a')]).toEqual([0, 1, 1, 0]);
    expect([...result.get('b')]).toEqual([10, 20, 10, 20]);
    expect([...result.get('c')]).toEqual([1, 2, 3, null]);
    expect([...result.get('c_R')]).toEqual([10, 30, null, 20]);
  });

  test('can join on multi-index with conflict, lsuffix', () => {
    const result =
      left_double.join({other: right_double_conflict, on: ['a', 'b'], how: 'outer', lsuffix: '_L'});
    expect(result.numColumns).toEqual(4);
    expect(result.names).toEqual(expect.arrayContaining(['a', 'b', 'c_L', 'c']));
    expect([...result.get('a')]).toEqual([0, 1, 1, 0]);
    expect([...result.get('b')]).toEqual([10, 20, 10, 20]);
    expect([...result.get('c_L')]).toEqual([1, 2, 3, null]);
    expect([...result.get('c')]).toEqual([10, 30, null, 20]);
  });

  test('can join on multi-index with conflict, lsuffix and rsuffix', () => {
    const result = left_double.join(
      {other: right_double_conflict, on: ['a', 'b'], how: 'outer', lsuffix: '_L', rsuffix: '_R'});
    expect(result.numColumns).toEqual(4);
    expect(result.names).toEqual(expect.arrayContaining(['a', 'b', 'c_L', 'c_R']));
    expect([...result.get('a')]).toEqual([0, 1, 1, 0]);
    expect([...result.get('b')]).toEqual([10, 20, 10, 20]);
    expect([...result.get('c_L')]).toEqual([1, 2, 3, null]);
    expect([...result.get('c_R')]).toEqual([10, 30, null, 20]);
  });

  test('can find common type for single-index join', () => {
    const left_double =
      new DataFrame({a: Series.new([1, 2, 3, 4, 5]), b: Series.new([0, 0, 1, 1, 2])});

    const result = left_double.join({other: right, on: ['b'], how: 'outer'});
    expect(result.numColumns).toEqual(3);
    expect(result.names).toEqual(expect.arrayContaining(['a', 'b', 'c']));
    expect([...result.get('b')]).toEqual([0, 0, 1, 1, 2, 3]);
    expect([...result.get('a')]).toEqual([1, 2, 3, 4, 5, null]);
    expect([...result.get('c')]).toEqual([0, 0, 10, 10, null, 30]);
  });

  test('can find common type on multi-index join', () => {
    const left_double_float = new DataFrame({
      a: Series.new([0, 1, 1]),
      b: Series.new([10, 20, 10]),
      c: Series.new({type: new Float32, data: [1., 2., 3.]})
    });

    const result = left_double_float.join({other: right_double, on: ['a', 'b'], how: 'outer'});
    expect(result.numColumns).toEqual(4);
    expect(result.names).toEqual(expect.arrayContaining(['a', 'b', 'c', 'd']));
    expect([...result.get('a')]).toEqual([0, 1, 1, 0]);
    expect([...result.get('b')]).toEqual([10, 20, 10, 20]);
    expect([...result.get('c')]).toEqual([1, 2, 3, null]);
    expect([...result.get('d')]).toEqual([10, 30, null, 20]);
  });
});

describe('DataFrame.join({how="right"}) ', () => {
  test('can join with no column name conflicts', () => {
    const result = left.join({other: right, on: ['b'], how: 'right'});
    expect(result.numColumns).toEqual(3);
    expect(result.names).toEqual(expect.arrayContaining(['a', 'b', 'c']));

    // Sorting is just to get 1-1 agreement with order of pd/cudf results
    const sorted_result = result.sortValues({b: {ascending: true, null_order: 'after'}});

    expect([...sorted_result.get('b')]).toEqual([0, 0, 1, 1, 3]);
    expect([...sorted_result.get('a')]).toEqual([1, 2, 3, 4, null]);
    expect([...sorted_result.get('c')]).toEqual([0, 0, 10, 10, 30]);
  });

  test('discards right conflicts without suffices', () => {
    const result = left.join({other: right_conflict, on: ['b'], how: 'right'});
    expect(result.numColumns).toEqual(2);
    expect(result.names).toEqual(expect.arrayContaining(['a', 'b']));

    // Sorting is just to get 1-1 agreement with order of pd/cudf results
    const sorted_result = result.sortValues({b: {ascending: true, null_order: 'after'}});

    expect([...sorted_result.get('a')]).toEqual([0, 0, 10, 10, 30]);
    expect([...sorted_result.get('b')]).toEqual([0, 0, 1, 1, 3]);
  });

  test('applies lsuffix', () => {
    const result = left.join({other: right_conflict, on: ['b'], how: 'right', lsuffix: '_L'});
    expect(result.numColumns).toEqual(3);
    expect(result.names).toEqual(expect.arrayContaining(['a', 'b', 'a_L']));

    // Sorting is just to get 1-1 agreement with order of pd/cudf results
    const sorted_result = result.sortValues({b: {ascending: true, null_order: 'after'}});

    expect([...sorted_result.get('b')]).toEqual([0, 0, 1, 1, 3]);
    expect([...sorted_result.get('a_L')]).toEqual([1, 2, 3, 4, null]);
    expect([...sorted_result.get('a')]).toEqual([0, 0, 10, 10, 30]);
  });

  test('applies rsuffix', () => {
    const result = left.join({other: right_conflict, on: ['b'], how: 'right', rsuffix: '_R'});
    expect(result.numColumns).toEqual(3);
    expect(result.names).toEqual(expect.arrayContaining(['a', 'b', 'a_R']));

    // Sorting is just to get 1-1 agreement with order of pd/cudf results
    const sorted_result = result.sortValues({b: {ascending: true, null_order: 'after'}});

    expect([...sorted_result.get('b')]).toEqual([0, 0, 1, 1, 3]);
    expect([...sorted_result.get('a')]).toEqual([1, 2, 3, 4, null]);
    expect([...sorted_result.get('a_R')]).toEqual([0, 0, 10, 10, 30]);
  });

  test('applies lsuffix and rsuffix', () => {
    const result =
      left.join({other: right_conflict, on: ['b'], how: 'right', lsuffix: '_L', rsuffix: '_R'});
    expect(result.numColumns).toEqual(3);
    expect(result.names).toEqual(expect.arrayContaining(['a_L', 'b', 'a_R']));

    // Sorting is just to get 1-1 agreement with order of pd/cudf results
    const sorted_result = result.sortValues({b: {ascending: true, null_order: 'after'}});

    expect([...sorted_result.get('b')]).toEqual([0, 0, 1, 1, 3]);
    expect([...sorted_result.get('a_L')]).toEqual([1, 2, 3, 4, null]);
    expect([...sorted_result.get('a_R')]).toEqual([0, 0, 10, 10, 30]);
  });

  test('can join on multi-index', () => {
    const result = left_double.join({other: right_double, on: ['a', 'b'], how: 'right'});
    expect(result.numColumns).toEqual(4);
    expect(result.names).toEqual(expect.arrayContaining(['a', 'b', 'c', 'd']));
    expect([...result.get('a')]).toEqual([0, 1, 0]);
    expect([...result.get('b')]).toEqual([10, 20, 20]);
    expect([...result.get('c')]).toEqual([1, 2, null]);
    expect([...result.get('d')]).toEqual([10, 30, 20]);
  });

  test('can find common type for single-index join', () => {
    const left_float =
      new DataFrame({a: Series.new([1, 2, 3, 4, 5]), b: Series.new([0, 0, 1, 1, 2])});

    const result = left_float.join({other: right, on: ['b'], how: 'right'});
    expect(result.numColumns).toEqual(3);
    expect(result.names).toEqual(expect.arrayContaining(['a', 'b', 'c']));

    // Sorting is just to get 1-1 agreement with order of pd/cudf results
    const sorted_result = result.sortValues({b: {ascending: true, null_order: 'after'}});

    expect([...sorted_result.get('b')]).toEqual([0, 0, 1, 1, 3]);
    expect([...sorted_result.get('a')]).toEqual([1, 2, 3, 4, null]);
    expect([...sorted_result.get('c')]).toEqual([0, 0, 10, 10, 30]);
  });

  test('can find common type on multi-index join', () => {
    const left_double_float = new DataFrame({
      a: Series.new([0, 1, 1]),
      b: Series.new([10, 20, 10]),
      c: Series.new({type: new Float32, data: [1., 2., 3.]})
    });

    const result = left_double_float.join({other: right_double, on: ['a', 'b'], how: 'right'});
    expect(result.numColumns).toEqual(4);
    expect(result.names).toEqual(expect.arrayContaining(['a', 'b', 'c', 'd']));
    expect([...result.get('a')]).toEqual([0, 1, 0]);
    expect([...result.get('b')]).toEqual([10, 20, 20]);
    expect([...result.get('c')]).toEqual([1, 2, null]);
    expect([...result.get('d')]).toEqual([10, 30, 20]);
  });
});

describe('DataFrame.join({how="leftsemi"}) ', () => {
  test('can semijoin with no column name conflicts', () => {
    const result = left.join({other: right, on: ['b'], how: 'leftanti'});
    expect(result.numColumns).toEqual(2);
    expect(result.names).toEqual(expect.arrayContaining(['a', 'b']));

    const sorted_result = result.sortValues({b: {ascending: true, null_order: 'after'}});

    expect([...sorted_result.get('b')]).toEqual([2]);
    expect([...sorted_result.get('a')]).toEqual([5]);
  });
});

describe('DataFrame.join({how="leftanti"}) ', () => {
  test('can antijoin with no column name conflicts', () => {
    const result = left.join({other: right, on: ['b'], how: 'leftsemi'});
    expect(result.numColumns).toEqual(2);
    expect(result.names).toEqual(expect.arrayContaining(['a', 'b']));

    const sorted_result = result.sortValues({b: {ascending: true, null_order: 'after'}});

    expect([...sorted_result.get('b')]).toEqual([0, 0, 1, 1]);
    expect([...sorted_result.get('a')]).toEqual([1, 2, 3, 4]);
  });
});
