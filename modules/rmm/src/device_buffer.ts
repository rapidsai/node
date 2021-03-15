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

import RMM from './addon';
import {MemoryResource} from './memory_resource';

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

/** @ignore */
export interface DeviceBufferConstructor {
  readonly prototype: DeviceBuffer;
  new(byteLength?: number, mr?: MemoryResource, stream?: number): DeviceBuffer;
  new(source?: DeviceBufferInput, mr?: MemoryResource, stream?: number): DeviceBuffer;
  new(sourceOrByteLength?: DeviceBufferInput|number, mr?: MemoryResource, stream?: number):
    DeviceBuffer;
}

/**
 * @summary Allocates device memory via RMM (the RAPIDS Memory Manager).
 * <br/><br/>
 * This class allocates untyped and *uninitialized* device memory using a
 * {@link MemoryResource}. If not explicitly specified, the memory resource
 * returned from {@link getCurrentDeviceResource} is used.
 */
export interface DeviceBuffer extends ArrayBuffer {
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
}

/** @ignore */
// eslint-disable-next-line @typescript-eslint/no-redeclare
export const DeviceBuffer: DeviceBufferConstructor = RMM.DeviceBuffer;
