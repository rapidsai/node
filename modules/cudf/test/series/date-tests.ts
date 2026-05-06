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

import {
  Int32Buffer,
  Int64Buffer,
  setDefaultAllocator,
} from '@rapidsai/cuda';
import {
  Series,
  TimestampDay,
  TimestampMicrosecond,
  TimestampMillisecond,
  TimestampNanosecond,
  TimestampSecond,
} from '@rapidsai/cudf';
import {DeviceBuffer} from '@rapidsai/rmm';

setDefaultAllocator((byteLength: number) => new DeviceBuffer(byteLength));

test('Series.TimestampDay (Int32Buffer)', () => {
  const dateTime = Math.floor(new Date('May 13, 2021 16:38:30:100 GMT+00:00').getTime() / 86400000);
  const s        = Series.new({type: new TimestampDay, data: new Int32Buffer([dateTime])});
  const val      = s.getValue(0);

  expect(val?.getUTCFullYear()).toBe(2021);
  expect(val?.getUTCMonth()).toBe(4);
  expect(val?.getUTCDate()).toBe(13);
});

test('Series.TimestampSecond (Int64Buffer)', () => {
  const dateTime = Math.floor(new Date('May 13, 2021 16:38:30:100 GMT+00:00').getTime() / 1000);
  const s = Series.new({type: new TimestampSecond, data: new Int64Buffer([dateTime].map(BigInt))});
  const val = s.getValue(0);

  expect(val?.getUTCFullYear()).toBe(2021);
  expect(val?.getUTCMonth()).toBe(4);
  expect(val?.getUTCDate()).toBe(13);
  expect(val?.getUTCHours()).toBe(16);
  expect(val?.getUTCMinutes()).toBe(38);
  expect(val?.getUTCSeconds()).toBe(30);
});

test('Series.TimestampMillisecond (Int64Buffer)', () => {
  const dateTime = new Date('May 13, 2021 16:38:30:100 GMT+00:00').getTime();
  const s =
    Series.new({type: new TimestampMillisecond, data: new Int64Buffer([dateTime].map(BigInt))});
  const val = s.getValue(0);

  expect(val?.getUTCFullYear()).toBe(2021);
  expect(val?.getUTCMonth()).toBe(4);
  expect(val?.getUTCDate()).toBe(13);
  expect(val?.getUTCHours()).toBe(16);
  expect(val?.getUTCMinutes()).toBe(38);
  expect(val?.getUTCSeconds()).toBe(30);
  expect(val?.getUTCMilliseconds()).toBe(100);
});

test('Series.TimestampNanosecond (Int64Buffer)', () => {
  const dateTime = new Date('May 13, 2021 16:38:30:100 GMT+00:00').getTime() * 1000;
  const s =
    Series.new({type: new TimestampMicrosecond, data: new Int64Buffer([dateTime].map(BigInt))});
  const val = s.getValue(0);

  expect(val?.getUTCFullYear()).toBe(2021);
  expect(val?.getUTCMonth()).toBe(4);
  expect(val?.getUTCDate()).toBe(13);
  expect(val?.getUTCHours()).toBe(16);
  expect(val?.getUTCMinutes()).toBe(38);
  expect(val?.getUTCSeconds()).toBe(30);
  expect(val?.getUTCMilliseconds()).toBe(100);
});

test('Series.TimestampMicrosecond (Int64Buffer)', () => {
  const dateTime = new Date('May 13, 2021 16:38:30:100 GMT+00:00').getTime() * 1000000;
  const s =
    Series.new({type: new TimestampNanosecond, data: new Int64Buffer([dateTime].map(BigInt))});
  const val = s.getValue(0);

  expect(val?.getUTCFullYear()).toBe(2021);
  expect(val?.getUTCMonth()).toBe(4);
  expect(val?.getUTCDate()).toBe(13);
  expect(val?.getUTCHours()).toBe(16);
  expect(val?.getUTCMinutes()).toBe(38);
  expect(val?.getUTCSeconds()).toBe(30);
  expect(val?.getUTCMilliseconds()).toBe(100);
});

test('Series.TimestampDay', () => {
  const dateTime = new Date('May 13, 2021 16:38:30:100 GMT+00:00');
  const s        = Series.new({type: new TimestampDay, data: [dateTime]});
  const val      = s.getValue(0);

  expect(val?.getUTCFullYear()).toBe(2021);
  expect(val?.getUTCMonth()).toBe(4);
  expect(val?.getUTCDate()).toBe(13);
});

test('Series.TimestampSecond', () => {
  const dateTime = new Date('May 13, 2021 16:38:30:100 GMT+00:00');
  const s        = Series.new({type: new TimestampSecond, data: [dateTime]});
  const val      = s.getValue(0);

  expect(val?.getUTCFullYear()).toBe(2021);
  expect(val?.getUTCMonth()).toBe(4);
  expect(val?.getUTCDate()).toBe(13);
  expect(val?.getUTCHours()).toBe(16);
  expect(val?.getUTCMinutes()).toBe(38);
  expect(val?.getUTCSeconds()).toBe(30);
});

test('Series.TimestampMillisecond', () => {
  const dateTime = new Date('May 13, 2021 16:38:30:100 GMT+00:00');
  const s        = Series.new({type: new TimestampMillisecond, data: [dateTime]});
  const val      = s.getValue(0);

  expect(val?.getUTCFullYear()).toBe(2021);
  expect(val?.getUTCMonth()).toBe(4);
  expect(val?.getUTCDate()).toBe(13);
  expect(val?.getUTCHours()).toBe(16);
  expect(val?.getUTCMinutes()).toBe(38);
  expect(val?.getUTCSeconds()).toBe(30);
  expect(val?.getUTCMilliseconds()).toBe(100);
});

test('Series.TimestampNanosecond', () => {
  const dateTime = new Date('May 13, 2021 16:38:30:100 GMT+00:00');
  const s        = Series.new({type: new TimestampMicrosecond, data: [dateTime]});
  const val      = s.getValue(0);

  expect(val?.getUTCFullYear()).toBe(2021);
  expect(val?.getUTCMonth()).toBe(4);
  expect(val?.getUTCDate()).toBe(13);
  expect(val?.getUTCHours()).toBe(16);
  expect(val?.getUTCMinutes()).toBe(38);
  expect(val?.getUTCSeconds()).toBe(30);
  expect(val?.getUTCMilliseconds()).toBe(100);
});

test('Series.TimestampMicrosecond', () => {
  const dateTime = new Date('May 13, 2021 16:38:30:100 GMT+00:00');
  const s        = Series.new({type: new TimestampNanosecond, data: [dateTime]});
  const val      = s.getValue(0);

  expect(val?.getUTCFullYear()).toBe(2021);
  expect(val?.getUTCMonth()).toBe(4);
  expect(val?.getUTCDate()).toBe(13);
  expect(val?.getUTCHours()).toBe(16);
  expect(val?.getUTCMinutes()).toBe(38);
  expect(val?.getUTCSeconds()).toBe(30);
  expect(val?.getUTCMilliseconds()).toBe(100);
});
