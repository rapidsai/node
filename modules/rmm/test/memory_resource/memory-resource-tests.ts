// Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const {comparable, supportsStreams, supportsGetMemInfo, createMemoryResource} = testConfig;

  test(`MemoryResource Constructor`, () => {
    let mr = createMemoryResource();
    expect(mr.supportsStreams).toEqual(supportsStreams);
    expect(mr.supportsGetMemInfo).toEqual(supportsGetMemInfo);
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

    const [freeStart, totalStart] = (mr.supportsGetMemInfo ? mr.getMemInfo() : [0, 0]);

    let dbuf = new DeviceBuffer(sizes['2_MiB'], mr);

    // Fill the buffer with 1s because managed memory is only allocated when it's actually used.
    new Uint8Buffer(dbuf).fill(1, 0, dbuf.byteLength);
    new Uint8Buffer(dbuf)[dbuf.byteLength - 1];

    const [freeEnd, totalEnd] = (mr.supportsGetMemInfo ? mr.getMemInfo() : [0, 0]);

    expect(totalStart).toEqual(totalEnd);

    if (mr.supportsGetMemInfo) { expect(freeStart - freeEnd).not.toEqual(0); }

    mr   = <any>null;
    dbuf = <any>null;
  });
});
