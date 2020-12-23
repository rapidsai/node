// Copyright (c) 2020, NVIDIA CORPORATION.
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
import {Uint8Buffer} from '@nvidia/cuda';
import {DeviceBuffer} from '@nvidia/rmm';
import {sizes, testForEachDevice} from '../utils';
import {memoryResourceTestConfigs} from './utils';

describe.each(memoryResourceTestConfigs)(`%s`, (_, testConfig) => {
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const {comparable, supportsStreams, supportsGetMemInfo, createMemoryResource} = testConfig;

  testForEachDevice(`construction`, () => {
    const mr = createMemoryResource();
    expect(mr.supportsStreams).toEqual(supportsStreams);
    expect(mr.supportsGetMemInfo).toEqual(supportsGetMemInfo);
  });

  testForEachDevice(`getMemInfo()`, () => {
    const mr         = createMemoryResource();
    const memoryInfo = mr.getMemInfo(0);
    expect(Array.isArray(memoryInfo)).toBe(true);
    expect(memoryInfo.length).toBe(2);
    memoryInfo.forEach((v) => expect(typeof v).toBe('number'));
  });

  testForEachDevice(`isEqual()`, () => {
    const mr1 = createMemoryResource();
    const mr2 = createMemoryResource();
    expect(mr1.isEqual(mr1)).toEqual(true);
    expect(mr1.isEqual(mr2)).toEqual(comparable);
    expect(mr2.isEqual(mr1)).toEqual(comparable);
    expect(mr2.isEqual(mr2)).toEqual(true);
  });

  testForEachDevice(`works with DeviceBuffer`, () => {
    const mr  = createMemoryResource();
    let free0 = 0, free1 = 0;
    let total0 = 0, total1 = 0;
    mr.supportsGetMemInfo && ([free0, total0] = mr.getMemInfo(0));
    // Fill the buffer with 1s, because CUDA Managed
    // memory is only allocated when it's actually used.
    // @ts-ignore
    let buf: Uint8Buffer|null = new Uint8Buffer(new DeviceBuffer(sizes ['2_MiB'], 0, mr)).fill(1);
    mr.supportsGetMemInfo && ([free1, total1] = mr.getMemInfo(0));
    expect(total0).toEqual(total1);
    if (mr.supportsGetMemInfo) { expect(free0 - free1).toBeGreaterThanOrEqual(sizes ['2_MiB']); }
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    buf = null;
  });
});
