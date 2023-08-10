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

import {Memory} from '@rapidsai/cuda';
import {MemoryResourceType} from './addon';

/** @ignore */
export declare const _cpp_exports: any;

export declare const per_device_resources: MemoryResource[];

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
 * @return The {@link MemoryResource} for the current device
 */
export declare function getCurrentDeviceResource(): MemoryResource;

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
 * @return The previous {@link MemoryResource} for the current device
 */
export declare function setCurrentDeviceResource(memoryResource: MemoryResource): MemoryResource;

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
export declare function getPerDeviceResource(deviceId: number): MemoryResource;

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
 * @return The previous {@link MemoryResource} for device `deviceId`
 */
export declare function setPerDeviceResource(deviceId: number,
                                             memoryResource: MemoryResource): MemoryResource;

export declare class MemoryResource {
  constructor(type: MemoryResourceType.CUDA, device?: number);
  constructor(type: MemoryResourceType.MANAGED);
  constructor(type: MemoryResourceType.POOL,
              upstreamMemoryResource: MemoryResource,
              initialPoolSize?: number,
              maximumPoolSize?: number);
  constructor(type: MemoryResourceType.FIXED_SIZE,
              upstreamMemoryResource: MemoryResource,
              blockSize?: number,
              blocksToPreallocate?: number);
  constructor(type: MemoryResourceType.BINNING,
              upstreamMemoryResource: MemoryResource,
              minSizeExponent?: number,
              maxSizeExponent?: number);
  constructor(type: MemoryResourceType.LOGGING,
              upstreamMemoryResource: MemoryResource,
              logFilePath?: string,
              autoFlush?: boolean);

  /**
   * @summary A boolean indicating whether the resource supports use of non-null CUDA streams for
   * allocation/deallocation.
   */
  readonly supportsStreams: boolean;

  /**
   * @summary A boolean indicating whether the resource supports the getMemInfo() API.
   */
  readonly supportsGetMemInfo: boolean;

  /**
   * Queries the amount of free and total memory for the resource.
   *
   * @param stream - the stream whose memory manager we want to retrieve
   *
   * @returns a tuple which contains `[free memory, total memory]` (in bytes)
   */
  getMemInfo(stream?: number): [number, number];

  /**
   * @summary Compare this resource to another.
   *
   * @remarks
   * Two `CudaMemoryResource` instances always compare equal, because they can each
   * deallocate memory allocated by the other.
   *
   * @param other - The other resource to compare to
   * @returns true if the two resources are equal, else false
   */
  isEqual(other: MemoryResource): boolean;
}

/** @ignore */
type FloatArray = Float32Array|Float64Array;
/** @ignore */
type IntArray = Int8Array|Int16Array|Int32Array;
/** @ignore */
type UintArray = Uint8Array|Uint16Array|Uint32Array|Uint8ClampedArray;
/** @ignore */
type BigIntArray = BigInt64Array|BigUint64Array;
/** @ignore */
type TypedArray = FloatArray|IntArray|UintArray;
/** @ignore */
type DeviceBufferInput = BigIntArray|TypedArray|ArrayBufferLike;

/**
 * @summary Allocates device memory via RMM (the RAPIDS Memory Manager).
 * <br/><br/>
 * This class allocates untyped and *uninitialized* device memory using a
 * {@link MemoryResource}. If not explicitly specified, the memory resource
 * returned from {@link getCurrentDeviceResource} is used.
 */
export declare class DeviceBuffer extends ArrayBuffer implements Memory {
  constructor(byteLength?: number, mr?: MemoryResource, stream?: number);
  constructor(source?: DeviceBufferInput, mr?: MemoryResource, stream?: number);
  constructor(sourceOrByteLength?: DeviceBufferInput|number, mr?: MemoryResource, stream?: number);
  /**
   * @summary The length in bytes of the {@link DeviceBuffer}.
   */
  readonly byteLength: number;
  /**
   * @summary Returns actual size in bytes of the device memory allocation.
   *
   * @note The invariant {@link byteLength} <= `capacity` holds.
   */
  readonly capacity: number;
  /**
   * @summary A boolean indicating whether the {@link byteLength} is zero.
   */
  readonly isEmpty: boolean;
  /** @ignore */
  readonly ptr: number;
  /**
   * @summary The CUDA Device associated with this {@link DeviceBuffer}.
   */
  readonly device: number;
  /**
   * @summary The CUDA stream most recently specified for allocation/deallocation.
   */
  readonly stream: number;
  /**
   * @summary The {@link MemoryResource} used to allocate and deallocate device memory.
   */
  readonly memoryResource: MemoryResource;

  /**
   * Resize the device memory allocation
   *
   * If the requested `new_size` is less than or equal to `capacity`, no
   * action is taken other than updating the value that is returned from
   * `byteLength`. Specifically, no memory is allocated nor copied. The value
   * `capacity` remains the actual size of the device memory allocation.
   *
   * @note {@link shrinkToFit} may be used to force the deallocation of unused
   * {@link capacity}.
   *
   * If `new_size` is larger than `capacity`, a new allocation is made on
   * `stream` to satisfy `new_size`, and the contents of the old allocation are
   * copied on `stream` to the new allocation. The old allocation is then freed.
   * The bytes from `[old_size, new_size)` are uninitialized.
   *
   * The invariant `byteLength <= capacity` holds.
   *
   * @param newSize - The requested new size, in bytes
   * @param stream - The stream to use for allocation and copy
   */
  resize(newSize: number, stream?: number): void;

  /**
   * Sets the stream to be used for deallocation
   *
   * If no other {@link DeviceBuffer} method that allocates or copies memory is
   * called after this call with a different stream argument, then `stream`
   * will be used for deallocation in the `{@link DeviceBuffer} destructor.
   * Otherwise, if another {@link DeviceBuffer} method with a stream parameter is
   * called after this, the later stream parameter will be stored and used in
   * the destructor.
   */
  setStream(stream: number): void;

  /**
   * Forces the deallocation of unused memory.
   *
   * Reallocates and copies on stream `stream` the contents of the device memory
   * allocation to reduce `capacity` to `byteLength`.
   *
   * If `byteLength == capacity`, no allocations nor copies occur.
   *
   * @param stream - The stream on which the allocation and copy are performed
   */
  shrinkToFit(stream: number): void;

  /**
   * Copy a slice of the current buffer into a new buffer
   *
   * @param start - the offset (in bytes) from which to start copying
   * @param end - the offset (in bytes) to stop copying, or the end of the
   * buffer if unspecified
   */
  slice(start: number, end?: number): DeviceBuffer;

  /**
   * @summary Explicitly free the device memory associated with this DeviceBuffer.
   */
  dispose(): void;
}
