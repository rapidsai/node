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

export * from './device_buffer';
export * from './memory_resource';

import RMM from './addon';
import {Device, devices} from '@nvidia/cuda';
import {MemoryResource} from './memory_resource';

const rmmSetPerDeviceResource: typeof setPerDeviceResource = RMM.setPerDeviceResource;
const perDeviceMemoryResources                             = new Map<number, MemoryResource>();

RMM.setPerDeviceResource = setPerDeviceResource;

export function getPerDeviceResource(deviceId: number) {
  if (!perDeviceMemoryResources.get(deviceId)) {
    devices [deviceId].callInContext(() =>
                                       setPerDeviceResource(deviceId, new RMM.MemoryResource(0)));
  }
  // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
  return perDeviceMemoryResources.get(deviceId)!;
}

export function setPerDeviceResource(
  deviceId: number, memoryResource: MemoryResource = getPerDeviceResource(deviceId)) {
  if (memoryResource && memoryResource !== perDeviceMemoryResources.get(deviceId)) {
    perDeviceMemoryResources.set(deviceId, memoryResource);
    rmmSetPerDeviceResource(deviceId, memoryResource);
  }
}

export function getPerDeviceResourceType(deviceId: number) {
  return getPerDeviceResource(deviceId)?.constructor;
}

export function getCurrentDeviceResource() { return getPerDeviceResource(Device.activeDeviceId); }

export function setCurrentDeviceResource(memoryResource: MemoryResource =
                                           getCurrentDeviceResource()) {
  return setPerDeviceResource(Device.activeDeviceId, memoryResource);
}
