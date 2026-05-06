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
import {Float64, Int32, Series} from '@rapidsai/cudf';
import {DeviceBuffer} from '@rapidsai/rmm';

setDefaultAllocator((byteLength: number) => new DeviceBuffer(byteLength));

test('NumericSeries.concat', () => {
  const col         = Series.new({type: new Int32, data: new Int32Buffer([1, 2, 3, 4, 5])});
  const colToConcat = Series.new({type: new Int32, data: new Int32Buffer([6, 7, 8, 9, 10])});

  const result = col.concat(colToConcat);

  expect([...result]).toEqual([...col, ...colToConcat]);
});

test('NumericSeries.concat up-casts to common dtype', () => {
  const col         = Series.new([1, 2, 3, 4, 5]).cast(new Int32);
  const colToConcat = Series.new([6, 7, 8, 9, 10]);

  const result = col.concat(colToConcat);

  expect(result.type).toBeInstanceOf(Float64);
  expect([...result]).toEqual([...col, ...colToConcat]);
});

test('Series.copy fixed width', () => {
  const col = Series.new({type: new Int32, data: new Int32Buffer([1, 2, 3, 4, 5])});

  const result = col.copy();

  expect([...result]).toEqual([...col]);
});

test('Series.copy String', () => {
  const col = Series.new(['foo', 'bar', 'test', null]);

  const result = col.copy();

  expect([...result]).toEqual([...col]);
});

test('Series.gather', () => {
  const col = Series.new({type: new Int32, data: new Int32Buffer([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])});

  const selection = Series.new({type: new Int32, data: new Int32Buffer([2, 4, 5, 8])});

  const result = col.gather(selection);

  expect([...result]).toEqual([...selection]);
});

test('Series.scatter (series)', () => {
  const col     = Series.new({type: new Int32, data: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]});
  const values  = Series.new({type: new Int32, data: [200, 400, 500, 800]});
  const indices = Series.new({type: new Int32, data: [2, 4, 5, 8]});

  const result = col.scatter(values, indices);

  expect([...result]).toEqual([0, 1, 200, 3, 400, 500, 6, 7, 800, 9]);
});

test('Series.scatter (series with array indices)', () => {
  const col     = Series.new({type: new Int32, data: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]});
  const values  = Series.new({type: new Int32, data: [200, 400, 500, 800]});
  const indices = [2, 4, 5, 8];

  const result = col.scatter(values, indices);

  expect([...result]).toEqual([0, 1, 200, 3, 400, 500, 6, 7, 800, 9]);
});

test('Series.scatter (scalar)', () => {
  const col     = Series.new({type: new Int32, data: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]});
  const indices = Series.new({type: new Int32, data: [2, 4, 5, 8]});

  const result = col.scatter(999, indices);

  expect([...result]).toEqual([0, 1, 999, 3, 999, 999, 6, 7, 999, 9]);
});

test('Series.scatter (scalar with array indicies)', () => {
  const col     = Series.new({type: new Int32, data: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]});
  const indices = [2, 4, 5, 8];

  const result = col.scatter(999, indices);

  expect([...result]).toEqual([0, 1, 999, 3, 999, 999, 6, 7, 999, 9]);
});

test('Series.scatter (scalar)', () => {
  const col     = Series.new({type: new Int32, data: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]});
  const indices = Series.new({type: new Int32, data: [2, 4, 5, 8]});

  const result = col.scatter(999, indices);

  expect([...result]).toEqual([0, 1, 999, 3, 999, 999, 6, 7, 999, 9]);
});

describe('Series.head', () => {
  const col = Series.new({type: new Int32, data: new Int32Buffer([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])});

  test('default n', () => { expect([...col.head()]).toEqual([0, 1, 2, 3, 4]); });

  test('invalid n', () => { expect(() => col.head(-1)).toThrowError(); });

  test('providing n', () => { expect([...col.head(8)]).toEqual([0, 1, 2, 3, 4, 5, 6, 7]); });

  test('n longer than length of series',
       () => { expect([...col.head(25)]).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]); });
});

describe('Series.tail', () => {
  const col = Series.new({type: new Int32, data: new Int32Buffer([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])});

  test('default n', () => { expect([...col.tail()]).toEqual([5, 6, 7, 8, 9]); });

  test('invalid n', () => { expect(() => col.tail(-1)).toThrowError(); });

  test('providing n', () => { expect([...col.tail(8)]).toEqual([2, 3, 4, 5, 6, 7, 8, 9]); });

  test('n longer than length of series',
       () => { expect([...col.tail(25)]).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]); });
});
