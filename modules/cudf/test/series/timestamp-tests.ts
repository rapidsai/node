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
import {
  Series,
  TimestampDay,
  TimestampMicrosecond,
  TimestampMillisecond,
  TimestampNanosecond,
  TimestampSecond,
} from '@rapidsai/cudf';
import {CudaMemoryResource, DeviceBuffer} from '@rapidsai/rmm';

const mr = new CudaMemoryResource();

setDefaultAllocator((byteLength: number) => new DeviceBuffer(byteLength, mr));

const data: Date[] = [
  new Date('2021-08-15T12:00:00.000Z'),
  new Date('2018-08-15T12:30:00.000Z'),
  new Date('2118-08-15T12:30:10.000Z'),
  new Date('2118-08-15T12:30:10.050Z'),
];

describe('TimestampNanosecond', () => {
  test('can create', () => {
    const col = Series.new({type: new TimestampNanosecond, data: data});
    expect([...col]).toEqual([
      1629028800000,
      1534336200000,
      4690009810000,
      4690009810050,
    ]);
  });
});

describe('TimestampMicrosecond', () => {
  test('can create', () => {
    const col = Series.new({type: new TimestampMicrosecond, data: data});
    expect([...col]).toEqual([
      1629028800000,
      1534336200000,
      4690009810000,
      4690009810050,
    ]);
  });
});

describe('TimestampMillisecond', () => {
  test('can create', () => {
    const col = Series.new({type: new TimestampMillisecond, data: data});
    expect([...col]).toEqual([
      1629028800000,
      1534336200000,
      4690009810000,
      4690009810050,
    ]);
  });
});

describe('TimestampSecond', () => {
  test('can create', () => {
    const col = Series.new({type: new TimestampSecond, data: data});
    expect([...col]).toEqual([
      1629028800000,
      1534336200000,
      4690009810000,
      4690009810000,  // millis truncated
    ]);
  });
});

describe('TimestampDay', () => {
  test('can create', () => {
    const col = Series.new({type: new TimestampDay, data: data});
    expect([...col]).toEqual([
      new Date('2021-08-15T00:00:00.000Z'),
      new Date('2018-08-15T00:00:00.000Z'),
      new Date('2118-08-15T00:00:00.000Z'),
      new Date('2118-08-15T00:00:00.000Z'),
    ]);
  });
});
