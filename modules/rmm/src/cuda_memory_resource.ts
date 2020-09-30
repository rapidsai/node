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

export interface CudaMemoryResourceConstructor {
    readonly prototype: CudaMemoryResource;
    new(): CudaMemoryResource;
}

export interface CudaMemoryResource {
    readonly supports_streams: boolean;

    readonly supports_get_mem_info: boolean;

   /**
   * Allocates memory of size at least `bytes`.
   *
   * The returned pointer will have at minimum 256 byte alignment.
   *
   * If supported, this operation may optionally be executed on a stream.
   * Otherwise, the stream is ignored and the null stream is used.
   *
   * @param bytes - The size of the allocation
   * @param stream - Stream on which to perform allocation
   * @returns Pointer to the newly allocated memory
   */
    allocate(bytes: number, stream?: number): number;

  /**
   * Deallocate memory pointed to by `ptr`.
   *
   * `ptr` must have been returned by a prior call to `allocate(bytes, stream)` on
   * a `device_memory_resource` that compares equal to this one, and the storage
   * it points to must not yet have been deallocated, otherwise behavior is
   * undefined.
   *
   * If supported, this operation may optionally be executed on a stream.
   * Otherwise, the stream is ignored and the null stream is used.
   *   *
   * @param ptr - Pointer to be deallocated
   * @param bytes - The size in bytes of the allocation. This must be equal to the
   * value of `bytes` that was passed to the `allocate` call that returned `p`.
   * @param stream - Stream on which to perform deallocation
   */
    deallocate(ptr: number, bytes: number, stream?: number): void;

    /**
    * Queries the amount of free and total memory for the resource.
    *
    * @param stream - the stream whose memory manager we want to retrieve
    *
    * @returns a tuple which contains `[free memory, total memory]` (in bytes)
    */
    get_mem_info(stream: number): [number, number];

   /**
   * Compare this resource to another.
   *
   * @remarks
   * Two `CudaMemoryResource` instances always compare equal, because they can each
   * deallocate memory allocated by the other.
   *
   * @param other - The other resource to compare to
   * @returns true if the two resources are equal, else false
   *
   */
    is_equal(other: CudaMemoryResource): boolean;
}

export const CudaMemoryResource: CudaMemoryResourceConstructor = RMM.CudaMemoryResource;