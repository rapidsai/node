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

export interface MemoryResourceConstructor {
  readonly prototype: MemoryResource;
  new (): MemoryResource;
}

export interface MemoryResource {
  /**
   * @summary A boolean indicating whether the resource supports use of non-null CUDA streams for allocation/deallocation.
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
  getMemInfo(stream: number): [number, number];

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

export interface CudaMemoryResourceConstructor {
  readonly prototype: CudaMemoryResource;

  /**
   * @summary Constructs a MemoryResource which allocates distinct chunks of CUDA GPU memory.
   * @param device The device ordinal on which to allocate memory (optional).
   */
  new (device?: number): CudaMemoryResource;
}

export type CudaMemoryResource = MemoryResource;

// eslint-disable-next-line @typescript-eslint/no-redeclare
export const CudaMemoryResource: CudaMemoryResourceConstructor = RMM.CudaMemoryResource;

export interface ManagedMemoryResourceConstructor {
  readonly prototype: ManagedMemoryResource;

  /**
   * @summary Constructs a MemoryResource which allocates distinct chunks of CUDA Managed memory.
   */
  new (): ManagedMemoryResource;
}

export type ManagedMemoryResource = MemoryResource;

// eslint-disable-next-line @typescript-eslint/no-redeclare
export const ManagedMemoryResource: ManagedMemoryResourceConstructor = RMM.ManagedMemoryResource;

export interface PoolMemoryResourceConstructor {
  readonly prototype: PoolMemoryResource;

  /**
   * @summary Constructs a coalescing best-fit suballocator which uses a pool of memory allocated from an upstream MemoryResource.
   * @param upstreamMemoryResource The MemoryResource from which to allocate blocks for the pool.
   * @param initialPoolSize Initial pool size in bytes. By default, an implementation-defined pool size is used.
   * @param maximumPoolSize Maximum size in bytes, that the pool can grow to.
   */
  new (
    upstreamMemoryResource: MemoryResource,
    initialPoolSize?: number,
    maximumPoolSize?: number,
  ): PoolMemoryResource;
}

export interface PoolMemoryResource extends MemoryResource {
  /**
   * @summary The MemoryResource from which to allocate blocks for the pool.
   */
  readonly upstreamMemoryResource: MemoryResource;
}

// eslint-disable-next-line @typescript-eslint/no-redeclare
export const PoolMemoryResource: PoolMemoryResourceConstructor = RMM.PoolMemoryResource;

export interface FixedSizeMemoryResourceConstructor {
  readonly prototype: FixedSizeMemoryResource;

  /**
   * @summary Constructs a MemoryResource which allocates memory blocks of a single fixed size from an upstream MemoryResource.
   * @param upstreamMemoryResource The MemoryResource from which to allocate blocks for the pool.
   * @param blockSize The size of blocks to allocate (default is 1MiB).
   * @param blocksToPreallocate The number of blocks to allocate to initialize the pool.
   */
  new (
    upstreamMemoryResource: MemoryResource,
    blockSize?: number,
    blocksToPreallocate?: number,
  ): FixedSizeMemoryResource;
}

export interface FixedSizeMemoryResource extends MemoryResource {
  /**
   * @summary The MemoryResource from which to allocate blocks for the pool.
   */
  readonly upstreamMemoryResource: MemoryResource;
}

// eslint-disable-next-line @typescript-eslint/no-redeclare
export const FixedSizeMemoryResource: FixedSizeMemoryResourceConstructor =
  RMM.FixedSizeMemoryResource;

export interface BinningMemoryResourceConstructor {
  readonly prototype: BinningMemoryResource;

  /**
   * @summary Constructs a MemoryResource which allocates memory from a set of specified "bin" sizes based on a specified allocation size from an upstream MemoryResource.
   *
   * @detail If minSizeExponent and maxSizeExponent are specified, initializes with one or more FixedSizeMemoryResource bins in the range [2^minSizeExponent, 2^maxSizeExponent].
   *
   * Call addBin to add additional bin allocators.
   *
   * @param upstreamMemoryResource The MemoryResource to use for allocations larger than any of the bins.
   * @param minSizeExponent The base-2 exponent of the minimum size FixedSizeMemoryResource bin to create (optional).
   * @param maxSizeExponent The base-2 exponent of the maximum size FixedSizeMemoryResource bin to create (optional).
   */
  new (
    upstreamMemoryResource: MemoryResource,
    minSizeExponent?: number,
    maxSizeExponent?: number,
  ): BinningMemoryResource;
}

export interface BinningMemoryResource extends MemoryResource {
  /**
   * The MemoryResource to use for allocations larger than any of the bins.
   */
  readonly upstreamMemoryResource: MemoryResource;

  /**
   * @summary Adds a bin of the specified maximum allocation size to this MemoryResource.
   *
   * @detail If specified, uses binResource for allocation for this bin. If not specified, uses a FixedSizeMemoryResource for this bin's allocations.
   *
   * Allocations smaller than byteLength and larger than the next smaller bin size will use this fixed-size MemoryResource.
   *
   * @param byteLength The maximum allocation size in bytes for the created bin
   * @param binResource The resource to use for this bin (optional)
   */
  addBin(byteLength: number, binResource?: MemoryResource): void;
}

// eslint-disable-next-line @typescript-eslint/no-redeclare
export const BinningMemoryResource: BinningMemoryResourceConstructor = RMM.BinningMemoryResource;

export interface LoggingResourceAdapterConstructor {
  readonly prototype: LoggingResourceAdapter;

  /**
   * @summary Constructs a MemoryResource that logs information about allocations/deallocations performed by an upstream MemoryResource.
   * @param upstreamMemoryResource The upstream MemoryResource to log.
   * @param logFilePath Path to the file to which logs are written. If not provided, falls back to the `RMM_LOG_FILE` environment variable.
   * @param autoFlush If true, flushes the log for every (de)allocation. Warning, this will degrade performance.
   */
  new (
    upstreamMemoryResource: MemoryResource,
    logFilePath?: string,
    autoFlush?: boolean,
  ): LoggingResourceAdapter;
}

export interface LoggingResourceAdapter extends MemoryResource {
  /**
   * Path to the file to which logs are written.
   */
  readonly logFilePath: string;

  /**
   * The MemoryResource to use for allocations larger than any of the bins.
   */
  readonly upstreamMemoryResource: MemoryResource;

  /**
   * @summary Flushes the buffered log contents.
   */
  flush(): void;
}

// eslint-disable-next-line @typescript-eslint/no-redeclare
export const LoggingResourceAdapter: LoggingResourceAdapterConstructor = RMM.LoggingResourceAdapter;
