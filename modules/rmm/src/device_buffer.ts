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

import RMM from './addon';
import {CudaMemoryResource} from './cuda_memory_resource';

export interface DeviceBufferConstructor {
    readonly prototype: DeviceBuffer;
    new(byteLength?: number, stream?: number, mr?: CudaMemoryResource): DeviceBuffer;
}

export interface DeviceBuffer {
    readonly byteLength: number;
    readonly capacity: number;
    readonly isEmpty: boolean;
    readonly ptr: number;
    readonly stream: number;

    /**
     * Resize the device memory allocation
     *
     * If the requested `new_size` is less than or equal to `capacity()`, no
     * action is taken other than updating the value that is returned from
     * `size()`. Specifically, no memory is allocated nor copied. The value
     * `capacity()` remains the actual size of the device memory allocation.
     *
     * @note `shrink_to_fit()` may be used to force the deallocation of unused
     * `capacity()`.
     *
     * If `new_size` is larger than `capacity()`, a new allocation is made on
     * `stream` to satisfy `new_size`, and the contents of the old allocation are
     * copied on `stream` to the new allocation. The old allocation is then freed.
     * The bytes from `[old_size, new_size)` are uninitialized.
     *
     * The invariant `size() <= capacity()` holds.
     *
     * @param newSize - The requested new size, in bytes
     * @param stream - The stream to use for allocation and copy
     */
    resize(newSize: number, stream?: number): void;

    /**
     * Sets the stream to be used for deallocation
     *
     * If no other rmm::device_buffer method that allocates or copies memory is
     * called after this call with a different stream argument, then @p stream
     * will be used for deallocation in the `rmm::device_buffer destructor.
     * Otherwise, if another rmm::device_buffer method with a stream parameter is
     * called after this, the later stream parameter will be stored and used in
     * the destructor.
     */
    setStream(stream: number): void;

    /**
     * Forces the deallocation of unused memory.
     *
     * Reallocates and copies on stream `stream` the contents of the device memory
     * allocation to reduce `capacity()` to `size()`.
     *
     * If `size() == capacity()`, no allocations nor copies occur.
     *
     * @param stream - The stream on which the allocation and copy are performed
     */
    shrinkToFit(stream: number): void;

    /**
     * Copy a slice of the current buffer into a new buffer
     *
     * @param begin - the offset (in bytes) to start copying from
     * @param end - the offset (in bytes) to end copying, or the end of the
     * buffer, if unspecified
     */
    slice(begin: number, end?: number): DeviceBuffer;
}

export const DeviceBuffer: DeviceBufferConstructor = RMM.DeviceBuffer;
