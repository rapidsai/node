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
import {devices} from '@nvidia/cuda';
import {
  DeviceBuffer,
  getPerDeviceResource,
  MemoryResource,
  setPerDeviceResource
} from '@rapidsai/rmm';

import {sizes} from '../utils';

import {memoryResourceTestConfigs} from './utils';

describe.each(memoryResourceTestConfigs)(`%s`, (_, {createMemoryResource}) => {
  test(`set/get per-device resource`, () => {
    const device                  = devices[0];
    let prev: MemoryResource|null = null;
    try {
      const mr = createMemoryResource();
      prev     = setPerDeviceResource(device.id, mr);
      expect(getPerDeviceResource(device.id)).toBe(mr);
      new DeviceBuffer(sizes['2_MiB'], mr);
    } finally {
      if (prev !== null) { setPerDeviceResource(device.id, prev); }
    }
  });
});
