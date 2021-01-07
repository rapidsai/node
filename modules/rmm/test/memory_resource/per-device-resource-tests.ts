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
import {Uint8Buffer} from '@nvidia/cuda';
import {DeviceBuffer, getPerDeviceResource, setPerDeviceResource} from '@nvidia/rmm';

import {sizes, testForEachDevice} from '../utils';

import {memoryResourceTestConfigs} from './utils';

describe.each(memoryResourceTestConfigs)(`%s`, (_, {createMemoryResource}) => {
  testForEachDevice(`set/get per-device resource`, (device) => {
    const currentMr = getPerDeviceResource(device.id);
    try {
      const mr = createMemoryResource();
      setPerDeviceResource(device.id, mr);
      expect(getPerDeviceResource(device.id)).toBe(mr);
      // Fill the buffer with 1s, because CUDA Managed
      // memory is only allocated when it's actually used.
      new Uint8Buffer(new DeviceBuffer(sizes ['2_MiB'], mr)).fill(1);
    } finally { setPerDeviceResource(device.id, currentMr); }
  });
});
