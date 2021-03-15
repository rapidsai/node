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

/**
 * @summary Get the {@link MemoryResource} for the specified device.
 *
 * <br/>
 *
 * Returns the {@link MemoryResource} set for the specified device. The initial resource is a {@link
 * CudaMemoryResource}.
 *
 * <br/>
 *
 * `deviceId` must be in the range `[0, Device.numDevices)`, otherwise behavior is
 * undefined.
 *
 * @note The returned {@link MemoryResource} should only be used when CUDA device `deviceId` is the
 * current device (e.g. set using `Device.activate()`). The behavior of a {@link MemoryResource}
 * is undefined if used while the active CUDA device is a different device from the one that was
 * active when the {@link MemoryResource} was created.
 *
 * @param deviceId The id of the target device
 * @return The current {@link MemoryResource} for device `deviceId`
 */
export function getPerDeviceResource(deviceId: number) {
  if (!perDeviceMemoryResources.get(deviceId)) {
    devices[deviceId].callInContext(() =>
                                      setPerDeviceResource(deviceId, new RMM.MemoryResource(0)));
  }
  // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
  return perDeviceMemoryResources.get(deviceId)!;
}

/**
 * @summary Set the {@link MemoryResource} for the specified device.
 *
 * If `memoryResource` is not `null`, sets the memory resource pointer for the device specified by
 * `deviceId` to `memoryResource`. If `memoryResource` is `null` or `undefined`, no action is taken.
 *
 * <br/>
 *
 * `deviceId` must be in the range `[0, Device.numDevices)`, otherwise behavior is
 * undefined.
 *
 * <br/>
 *
 * The `memoryResource` must outlive the last use of the resource, otherwise behavior
 * is undefined. It is the caller's responsibility to maintain the lifetime of the
 * `memoryResource` object.
 *
 * @note The supplied `memoryResource` must have been created for the current CUDA device. The
 * behavior of a {@link MemoryResource} is undefined if used while the active CUDA device is a
 * different device from the one that was active when the {@link MemoryResource} was created.
 *
 * @param deviceId The id of the target device
 * @param memoryResource If not `null`, the new {@link MemoryResource} to use for device `deviceId`.
 */
export function setPerDeviceResource(
  deviceId: number, memoryResource: MemoryResource = getPerDeviceResource(deviceId)) {
  if (memoryResource && memoryResource !== perDeviceMemoryResources.get(deviceId)) {
    perDeviceMemoryResources.set(deviceId, memoryResource);
    rmmSetPerDeviceResource(deviceId, memoryResource);
  }
}

/**
 * @summary Get the {@link MemoryResource} for the specified device.
 *
 * <br/>
 *
 * Returns the {@link MemoryResource} for the specified device. The initial resource is a {@link
 * CudaMemoryResource}.
 *
 * <br/>
 *
 * `deviceId` must be in the range `[0, Device.numDevices)`, otherwise behavior is
 * undefined.
 *
 * @note The returned {@link MemoryResource} should only be used when CUDA device `deviceId` is the
 * current device (e.g. set using `Device.activate()`). The behavior of a {@link MemoryResource}
 * is undefined if used while the active CUDA device is a different device from the one that was
 * active when the {@link MemoryResource} was created.
 *
 * @param id The id of the target device
 * @return The current {@link MemoryResource} for device `deviceId`
 */
export function getPerDeviceResourceType(deviceId: number) {
  return getPerDeviceResource(deviceId)?.constructor;
}

/**
 * @summary Get the {@link MemoryResource} for the current device.
 *
 * <br/>
 *
 * Returns the {@link MemoryResource} set for the current device. The initial resource is a {@link
 * CudaMemoryResource}.
 *
 * <br/>
 *
 * The "current device" is the device specified by {@link Device.activeDeviceId}.
 *
 * <br/>
 *
 * @note The returned {@link MemoryResource} should only be used with the current CUDA device.
 * Changing the current device (e.g. calling `Device.activate()`) and then using the returned
 * resource can result in undefined behavior. The behavior of a {@link MemoryResource} is undefined
 * if used while the active CUDA device is a different device from the one that was active when the
 * {@link MemoryResource} was created.
 *
 * @return Pointer to the resource for the current device
 */
export function getCurrentDeviceResource() { return getPerDeviceResource(Device.activeDeviceId); }

/**
 * @summary Set the memory resource for the current device.
 *
 * <br/>
 *
 * If `memoryResource` is not `null`, sets the {@link MemoryResource} for the current device to
 * `memoryResource`. If `memoryResource` is `null` or `undefined`, no action is taken.
 *
 * <br/>
 *
 * The "current device" is the device specified by {@link Device.activeDeviceId}.
 *
 * <br/>
 *
 * The `memoryResource` must outlive the last use of the resource, otherwise behavior
 * is undefined. It is the caller's responsibility to maintain the lifetime of the  {@link
 * MemoryResource `memoryResource`} object.
 *
 * <br/>
 *
 * @note The supplied `memoryResource` must have been created for the current CUDA device. The
 * behavior of a {@link MemoryResource} is undefined if used while the active CUDA device is a
 * different device from the one that was active when the {@link MemoryResource} was created.
 *
 * @param memoryResource If not `null`, the new resource to use for the current device.
 */

export function setCurrentDeviceResource(memoryResource: MemoryResource =
                                           getCurrentDeviceResource()) {
  return setPerDeviceResource(Device.activeDeviceId, memoryResource);
}
