// Copyright (c) 2020-2026, NVIDIA CORPORATION.
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

import {expect} from '@jest/globals';
import {Uint8Buffer} from '@rapidsai/cuda';
import {DeviceBuffer} from '@rapidsai/rmm';

import {sizes} from '../utils';

import {memoryResourceTestConfigs} from './utils';

describe.each(memoryResourceTestConfigs)(`%s`, (_, testConfig) => {
  const {comparable, createMemoryResource} = testConfig;

  test(`MemoryResource Constructor`, () => {
    let mr = createMemoryResource();
    // Basic construction test - just ensure the resource is created
    expect(mr).toBeDefined();
    mr = <any>null;
  });

  test(`MemoryResource.prototype.getMemInfo`, () => {
    let mr           = createMemoryResource();
    const memoryInfo = mr.getMemInfo();
    expect(Array.isArray(memoryInfo)).toBe(true);
    expect(memoryInfo.length).toBe(2);
    memoryInfo.forEach((v) => expect(typeof v).toBe('number'));
    mr = <any>null;
  });

  test(`MemoryResource.prototype.isEqual`, () => {
    let mr1 = createMemoryResource();
    let mr2 = createMemoryResource();
    expect(mr1.isEqual(mr1)).toEqual(true);
    expect(mr1.isEqual(mr2)).toEqual(comparable);
    expect(mr2.isEqual(mr1)).toEqual(comparable);
    expect(mr2.isEqual(mr2)).toEqual(true);
    mr1 = <any>null;
    mr2 = <any>null;
  });

  test(`works with DeviceBuffer`, () => {
    let mr = createMemoryResource();

    let dbuf = new DeviceBuffer(sizes['2_MiB'], mr);

    // Fill the buffer with 1s because managed memory is only allocated when it's actually used.
    new Uint8Buffer(dbuf).fill(1, 0, dbuf.byteLength);
    new Uint8Buffer(dbuf)[dbuf.byteLength - 1];

    // Basic test - just ensure the buffer was created and filled successfully
    expect(dbuf.byteLength).toEqual(sizes['2_MiB']);

    mr   = <any>null;
    dbuf = <any>null;
  });
});
