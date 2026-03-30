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

import {Float32Buffer, Int32Buffer, setDefaultAllocator} from '@rapidsai/cuda';
import {Float32, Int32, Series} from '@rapidsai/cudf';
import {DeviceBuffer} from '@rapidsai/rmm';
import * as arrow from 'apache-arrow';

setDefaultAllocator((byteLength: number) => new DeviceBuffer(byteLength));

test('Series.filter', () => {
  const col = Series.new([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);

  const mask = Series.new([false, false, true, false, true, true, false, false, true, false]);

  const result = col.filter(mask);

  const expected = Series.new({type: new Int32, data: new Int32Buffer([2, 4, 5, 8])});
  expect([...result]).toEqual([...expected]);
});

describe('Series.nLargest', () => {
  const col =
    Series.new({type: new Int32, data: new Int32Buffer([9, 5, 0, 2, 1, 3, 4, 7, 6, 8, 0])});

  test('default n', () => { expect([...col.nLargest()]).toEqual([9, 8, 7, 6, 5]); });

  test('negative n', () => { expect([...col.nLargest(-1)]).toEqual([]); });

  test('providing n', () => { expect([...col.nLargest(8)]).toEqual([9, 8, 7, 6, 5, 4, 3, 2]); });

  test('n longer than length of series',
       () => { expect([...col.nLargest(25)]).toEqual([9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0]); });

  test('keep last duplicate option', () => {
    expect([...col.nLargest(25, 'last')]).toEqual([9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0]);

    expect([...col.nLargest(-5, 'last')]).toEqual([]);
  });

  test('keep none duplicate option throws',
       () => { expect(() => col.nLargest(25, 'none')).toThrow(); });
});

describe('Series.nSmallest', () => {
  const col =
    Series.new({type: new Int32, data: new Int32Buffer([9, 5, 0, 2, 1, 3, 4, 7, 6, 8, 0])});

  test('default n', () => { expect([...col.nSmallest()]).toEqual([0, 0, 1, 2, 3]); });

  test('negative n', () => { expect([...col.nSmallest(-1)]).toEqual([]); });

  test('providing n', () => { expect([...col.nSmallest(8)]).toEqual([0, 0, 1, 2, 3, 4, 5, 6]); });

  test('n longer than length of series',
       () => { expect([...col.nSmallest(25)]).toEqual([0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]); });

  test('keep last duplicate option', () => {
    expect([...col.nSmallest(25, 'last')]).toEqual([0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);

    expect([...col.nSmallest(-5, 'last')]).toEqual([]);
  });

  test('keep none duplicate option throws',
       () => { expect(() => col.nSmallest(25, 'none')).toThrow(); });
});

test('Series.isNull (numeric)', () => {
  const col    = Series.new({type: new Int32, data: [0, 1, null, 3, 4, null, 6, null]});
  const result = col.isNull();

  const expected = [false, false, true, false, false, true, false, true];
  expect([...result]).toEqual(expected);
});

test('Series.isNotNull (numeric)', () => {
  const col    = Series.new({type: new Int32, data: [0, 1, null, 3, 4, null, 6, null]});
  const result = col.isNotNull();

  const expected = [true, true, false, true, true, false, true, false];
  expect([...result]).toEqual(expected);
});

test('Series.dropNulls (drop nulls only)', () => {
  const mask = arrow.vectorFromArray([0, 1, 1, 1, 1, 0], new arrow.Bool).data[0].values;
  const col =
    Series.new({type: new Float32, data: new Float32Buffer([1, 3, NaN, 4, 2, 0]), nullMask: mask});
  const result = col.dropNulls();

  const expected = [3, NaN, 4, 2];
  expect([...result]).toEqual(expected);
});

test('FloatSeries.dropNaNs (drop NaN values only)', () => {
  const mask = arrow.vectorFromArray([0, 1, 1, 1, 1, 0], new arrow.Bool).data[0].values;
  const col =
    Series.new({type: new Float32, data: new Float32Buffer([1, 3, NaN, 4, 2, 0]), nullMask: mask});
  const result = col.dropNaNs();

  const expected = [null, 3, 4, 2, null];
  expect([...result]).toEqual(expected);
});

test('Series.countNonNulls', () => {
  const twoNonNulls  = Series.new(['foo', null, 'bar']);
  const fourNonNulls = Series.new([NaN, null, 10, 15, 17, null]);
  const fiveNonNulls = Series.new([0, 1, null, 3, 4, null, 6, null]);

  expect(twoNonNulls.countNonNulls()).toEqual(2);
  expect(fourNonNulls.countNonNulls()).toEqual(4);
  expect(fiveNonNulls.countNonNulls()).toEqual(5);
});

test('FloatSeries.nansToNulls', () => {
  const col = Series.new({type: new Float32, data: new Float32Buffer([1, 3, NaN, 4, 2, 0])});

  const result = col.nansToNulls();

  const expected = [1, 3, null, 4, 2, 0];
  expect([...result]).toEqual(expected);
  expect(result.nullCount).toEqual(1);
  expect(col.nullCount).toEqual(0);
});

test('Series.valueCounts', () => {
  const s      = Series.new({type: new Int32, data: [110, 120, 100, 110, 120, 120]});
  const result = s.valueCounts();
  const count  = [...result.count];
  const value  = [...result.value];

  const countMap: Record<number, number> = {100: 1, 110: 2, 120: 3};

  for (let i = 0; i < value.length; i++) {
    const currentVal   = value[i] as number;
    const currentCount = count[i];
    expect(currentCount).toBe(countMap[currentVal]);
  }
});

test.each`
  nullsEqual | data                           | expected
  ${true}    | ${[null, null, 1, 2, 3, 4, 4]} | ${[null, 1, 2, 3, 4]}
  ${false}   | ${[null, null, 1, 2, 3, 4, 4]} | ${[null, null, 1, 2, 3, 4]}
`('Series.unique($nullsEqual)', ({nullsEqual, data, expected}) => {
  const s      = Series.new({type: new Int32, data});
  const result = s.unique(nullsEqual);
  expect([...result]).toEqual(expected);
});

test.each`
  data                    | replaceValue                          | expected
  ${[1, null, 3]}         | ${Series.new([9, 9, 9])}              | ${[1, 9, 3]}
  ${['foo', 'bar', null]} | ${Series.new(['test','test','test'])} | ${['foo', 'bar', 'test']}
  ${[true, false, null]}  | ${Series.new([false, false, false])}  | ${[true, false, false]}
  ${[1, null, 3]}         | ${9}                                  | ${[1, 9, 3]}
  ${['foo', 'bar', null]} | ${'test'}                             | ${['foo', 'bar', 'test']}
  ${[true, false, null]}  | ${false}                              | ${[true, false, false]}
`('Series.replaceNulls', ({data, replaceValue, expected}) => {
  const s       = Series.new(data);
  const result  = s.replaceNulls(replaceValue);

  expect([...result]).toEqual(expected);
});

test.each`
  data                    | expected
  ${[1, null, 3]}         | ${[1, 1, 3]}
  ${['foo', 'bar', null]} | ${['foo', 'bar', 'bar']}
  ${[true, false, null]}  | ${[true, false, false]}
`('Series.replaceNullsPreceding', ({data, expected})=> {
  const s       = Series.new(data);
  const result  = s.replaceNullsPreceding();

  expect([...result]).toEqual(expected);
});

test.each`
                   data | expected
        ${[1, null, 3]} | ${[1, 3, 3]}
${['foo', 'bar', null]} | ${['foo', 'bar', null]}
  ${[true, null, true]} | ${[true, true, true]}
`('Series.replaceNullsFollowing', ({data, expected})=> {
  const s       = Series.new(data);
  const result  = s.replaceNullsFollowing();

  expect([...result]).toEqual(expected);
});

test.each`
  keep     | nullsEqual | nullsFirst | data                           | expected
  ${true}  | ${true}    | ${true}    | ${[4, null, 1, 2, null, 3, 4]} | ${[null, 1, 2, 3, 4]}
  ${false} | ${true}    | ${true}    | ${[4, null, 1, 2, null, 3, 4]} | ${[1, 2, 3]}
  ${true}  | ${true}    | ${false}   | ${[4, null, 1, 2, null, 3, 4]} | ${[1, 2, 3, 4, null]}
  ${false} | ${true}    | ${false}   | ${[4, null, 1, 2, null, 3, 4]} | ${[1, 2, 3]}
  ${true}  | ${false}   | ${true}    | ${[4, null, 1, 2, null, 3, 4]} | ${[null, null, 1, 2, 3, 4]}
  ${false} | ${false}   | ${true}    | ${[4, null, 1, 2, null, 3, 4]} | ${[null, null, 1, 2, 3]}
  ${true}  | ${false}   | ${false}   | ${[4, null, 1, 2, null, 3, 4]} | ${[1, 2, 3, 4, null, null]}
  ${false} | ${false}   | ${false}   | ${[4, null, 1, 2, null, 3, 4]} | ${[1, 2, 3, null, null]}
`('Series.dropDuplicates($keep, $nullsEqual, $nullsFirst)', ({keep, nullsEqual, nullsFirst, data, expected}) => {
  const s      = Series.new({type: new Int32, data});
  const result = s.dropDuplicates(keep, nullsEqual, nullsFirst);
  expect([...result]).toEqual(expected);
});
