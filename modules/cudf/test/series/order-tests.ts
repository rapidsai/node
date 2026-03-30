// Copyright (c) 2026, NVIDIA CORPORATION.
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

import {Int32Buffer, setDefaultAllocator} from '@rapidsai/cuda';
import {Int32, Series} from '@rapidsai/cudf';
import {DeviceBuffer} from '@rapidsai/rmm';
import * as arrow from 'apache-arrow';

setDefaultAllocator((byteLength: number) => new DeviceBuffer(byteLength));

test('Series.orderBy (ascending, non-null)', () => {
  const col    = Series.new({type: new Int32, data: new Int32Buffer([1, 3, 5, 4, 2, 0])});
  const result = col.orderBy(true, 'before');

  const expected = [5, 0, 4, 1, 3, 2];
  expect([...result]).toEqual(expected);
});

test('Series.orderBy (descending, non-null)', () => {
  const col    = Series.new({type: new Int32, data: new Int32Buffer([1, 3, 5, 4, 2, 0])});
  const result = col.orderBy(false, 'before');

  const expected = [2, 3, 1, 4, 0, 5];
  expect([...result]).toEqual(expected);
});

test('Series.orderBy (ascending, null before)', () => {
  const mask = arrow.vectorFromArray([1, 0, 1, 1, 1, 1], new arrow.Bool).data[0].values;
  const col =
    Series.new({type: new Int32, data: new Int32Buffer([1, 3, 5, 4, 2, 0]), nullMask: mask});
  const result = col.orderBy(true, 'before');

  const expected = [1, 5, 0, 4, 3, 2];
  expect([...result]).toEqual(expected);
});

test('Series.orderBy (ascending, null after)', () => {
  const mask = arrow.vectorFromArray([1, 0, 1, 1, 1, 1], new arrow.Bool).data[0].values;
  const col =
    Series.new({type: new Int32, data: new Int32Buffer([1, 3, 5, 4, 2, 0]), nullMask: mask});
  const result = col.orderBy(true, 'after');

  const expected = [5, 0, 4, 3, 2, 1];
  expect([...result]).toEqual(expected);
});

test('Series.orderBy (descendng, null before)', () => {
  const mask = arrow.vectorFromArray([1, 0, 1, 1, 1, 1], new arrow.Bool).data[0].values;
  const col =
    Series.new({type: new Int32, data: new Int32Buffer([1, 3, 5, 4, 2, 0]), nullMask: mask});
  const result = col.orderBy(false, 'before');

  const expected = [2, 3, 4, 0, 5, 1];

  expect([...result]).toEqual(expected);
});

test('Series.orderBy (descending, null after)', () => {
  const mask = arrow.vectorFromArray([1, 0, 1, 1, 1, 1], new arrow.Bool).data[0].values;
  const col =
    Series.new({type: new Int32, data: new Int32Buffer([1, 3, 5, 4, 2, 0]), nullMask: mask});
  const result = col.orderBy(false, 'after');

  const expected = [1, 2, 3, 4, 0, 5];
  expect([...result]).toEqual(expected);
});

test('Series.sortValues (ascending)', () => {
  const col    = Series.new({type: new Int32, data: new Int32Buffer([1, 3, 5, 4, 2, 0])});
  const result = col.sortValues();

  const expected = [0, 1, 2, 3, 4, 5];
  expect([...result]).toEqual(expected);
});

test('Series.sortValues (descending)', () => {
  const col    = Series.new({type: new Int32, data: new Int32Buffer([1, 3, 5, 4, 2, 0])});
  const result = col.sortValues(false);

  const expected = [5, 4, 3, 2, 1, 0];
  expect([...result]).toEqual(expected);
});

test('Series.reverse', () => {
  const array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
  const col   = Series.new(array);

  expect([...col.reverse()]).toEqual(array.reverse());
});
