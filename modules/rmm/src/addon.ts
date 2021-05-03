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

/* eslint-disable @typescript-eslint/no-redeclare */

import {addon as CUDA} from '@nvidia/cuda';
import {loadNativeModule} from '@rapidsai/core';

export const {
  DeviceBuffer,
  MemoryResource,
  getPerDeviceResource,
  setPerDeviceResource,
  getCurrentDeviceResource,
  setCurrentDeviceResource,
  _cpp_exports,
  per_device_resources
} = loadNativeModule<typeof import('./node_rmm')>(module, 'node_rmm', init => init(CUDA));

export type DeviceBuffer   = import('./node_rmm').DeviceBuffer;
export type MemoryResource = import('./node_rmm').MemoryResource;

export type setPerDeviceResource = typeof import('./node_rmm').setPerDeviceResource;

export const enum MemoryResourceType
{
  CUDA      = 0,
  MANAGED   = 1,
  POOL      = 2,
  FIXEDSIZE = 3,
  BINNING   = 4,
  LOGGING   = 5,
}
