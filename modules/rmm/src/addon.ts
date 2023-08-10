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

import {addon as CORE} from '@rapidsai/core';
import {addon as CUDA} from '@rapidsai/cuda';

export const {
  DeviceBuffer,
  MemoryResource,
  getPerDeviceResource,
  setPerDeviceResource,
  getCurrentDeviceResource,
  setCurrentDeviceResource,
  _cpp_exports,
  per_device_resources
} = require('bindings')('rapidsai_rmm.node').init(CORE, CUDA) as typeof import('./node_rmm');

export type DeviceBuffer   = import('./node_rmm').DeviceBuffer;
export type MemoryResource = import('./node_rmm').MemoryResource;

export type setPerDeviceResource = typeof import('./node_rmm').setPerDeviceResource;

export const enum MemoryResourceType {
  /* ALIGNED_ADAPTOR          = 0, */
  /* ARENA                    = 1, */
  BINNING = 2,
  /* CUDA_ASYNC               = 3, */
  CUDA = 4,
  /* DEVICE                   = 5, */
  FIXED_SIZE = 6,
  /* LIMITING_ADAPTOR         = 7, */
  LOGGING = 8,
  MANAGED = 9,
  /* POLYMORPHIC_ALLOCATOR    = 10, */
  POOL = 11,
  /* STATISTICS_ADAPTOR       = 12, */
  /* THREAD_SAFE_ADAPTOR      = 13, */
  /* THRUST_ALLOCATOR_ADAPTOR = 14, */
  /* TRACKING_ADAPTOR         = 15, */
}
