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

import {MemoryResource, MemoryResourceType} from './addon';

/** @ignore */
export {MemoryResource, MemoryResourceType};

/**
 * @summary {@link MemoryResource} that uses cudaMalloc/Free for allocation/deallocation.
 */
export class CudaMemoryResource extends MemoryResource {
  /**
   * @summary Constructs a MemoryResource which allocates distinct chunks of CUDA GPU memory.
   * @param device The device ordinal on which to allocate memory (optional).
   */
  constructor(device?: number) { super(MemoryResourceType.CUDA, device); }
}

/**
 * @summary {@link MemoryResource} that uses cudaMallocManaged/Free for allocation/deallocation.
 */
export class ManagedMemoryResource extends MemoryResource {
  /**
   * @summary Constructs a MemoryResource which allocates distinct chunks of CUDA Managed memory.
   */
  constructor() { super(MemoryResourceType.MANAGED); }
}

export interface PoolMemoryResource extends MemoryResource {
  /**
   * @summary The {@link MemoryResource} from which to allocate blocks for the pool.
   */
  readonly memoryResource: MemoryResource;
}

/**
 * @summary A coalescing best-fit suballocator which uses a pool of memory allocated from an
 * upstream {@link MemoryResource}.
 */
export class PoolMemoryResource extends MemoryResource {
  /**
   * @summary Constructs a coalescing best-fit suballocator which uses a pool of memory allocated
   * from an upstream MemoryResource.
   * @param upstreamMemoryResource The MemoryResource from which to allocate blocks for the pool.
   * @param initialPoolSize Initial pool size in bytes. By default, an implementation-defined pool
   *   size is used.
   * @param maximumPoolSize Maximum size in bytes, that the pool can grow to.
   */
  constructor(upstreamMemoryResource: MemoryResource,
              initialPoolSize?: number,
              maximumPoolSize?: number) {
    super(MemoryResourceType.POOL, upstreamMemoryResource, initialPoolSize, maximumPoolSize);
  }
}

export interface FixedSizeMemoryResource extends MemoryResource {
  /**
   * @summary The {@link MemoryResource} from which to allocate blocks for the pool.
   */
  readonly memoryResource: MemoryResource;
}

/**
 * @summary A {@link MemoryResource} which allocates memory blocks of a single fixed size using an
 * upstream {@link MemoryResource}.
 *
 * @note Supports only allocations of size smaller than the configured `blockSize`.
 */
export class FixedSizeMemoryResource extends MemoryResource {
  /**
   * @summary Constructs a MemoryResource which allocates memory blocks of a single fixed size using
   * an upstream {@link MemoryResource}.
   * @param upstreamMemoryResource The {@link MemoryResource} from which to allocate blocks for the
   *   pool.
   * @param blockSize The size of blocks to allocate (default is 1MiB).
   * @param blocksToPreallocate The number of blocks to allocate to initialize the pool.
   */
  constructor(upstreamMemoryResource: MemoryResource,
              blockSize?: number,
              blocksToPreallocate?: number) {
    super(MemoryResourceType.FIXEDSIZE, upstreamMemoryResource, blockSize, blocksToPreallocate);
  }
}

export interface BinningMemoryResource extends MemoryResource {
  /**
   * The {@link MemoryResource} to use for allocations larger than any of the bins.
   */
  readonly memoryResource: MemoryResource;

  /**
   * @summary Adds a bin of the specified maximum allocation size to this MemoryResource.
   *
   * @detail If specified, uses binResource for allocation for this bin. If not specified, uses a
   * FixedSizeMemoryResource for this bin's allocations.
   *
   * Allocations smaller than byteLength and larger than the next smaller bin size will use this
   * fixed-size MemoryResource.
   *
   * @param byteLength The maximum allocation size in bytes for the created bin
   * @param binResource The resource to use for this bin (optional)
   */
  addBin(byteLength: number, binResource?: MemoryResource): void;
}

/**
 * @summary Allocates memory from upstream {@link MemoryResource resources} associated with bin
 * sizes.
 */
export class BinningMemoryResource extends MemoryResource {
  /**
   * @summary Constructs a {@link MemoryResource} which allocates memory from a set of specified
   * "bin" sizes based on a specified allocation size from an upstream {@link MemoryResource}.
   *
   * @detail If minSizeExponent and maxSizeExponent are specified, initializes with one or more
   * FixedSizeMemoryResource bins in the range [2^minSizeExponent, 2^maxSizeExponent].
   *
   * Call addBin to add additional bin allocators.
   *
   * @param upstreamMemoryResource The MemoryResource to use for allocations larger than any of the
   *   bins.
   * @param minSizeExponent The base-2 exponent of the minimum size FixedSizeMemoryResource bin to
   *   create (optional).
   * @param maxSizeExponent The base-2 exponent of the maximum size FixedSizeMemoryResource bin to
   *   create (optional).
   */
  constructor(upstreamMemoryResource: MemoryResource,
              minSizeExponent?: number,
              maxSizeExponent?: number) {
    super(MemoryResourceType.BINNING, upstreamMemoryResource, minSizeExponent, maxSizeExponent);
  }
}

export interface LoggingResourceAdapter extends MemoryResource {
  /**
   * Path to the file to which logs are written.
   */
  readonly logFilePath: string;

  /**
   * The MemoryResource to use for allocations larger than any of the bins.
   */
  readonly memoryResource: MemoryResource;

  /**
   * @summary Flushes the buffered log contents.
   */
  flush(): void;
}

/**
 * @brief Resource that uses an upstream {@link MemoryResource} to allocate memory and logs
 * information about the requested allocation/deallocations.
 * <br/><br/>
 * An instance of this resource can be constructed with an existing, upstream resource in order to
 * satisfy allocation requests and log allocation/deallocation activity.
 */
export class LoggingResourceAdapter extends MemoryResource {
  /**
   * @summary Constructs a MemoryResource that logs information about allocations/deallocations
   * performed by an upstream MemoryResource.
   * @param upstreamMemoryResource The upstream MemoryResource to log.
   * @param logFilePath Path to the file to which logs are written. If not provided, falls back to
   *   the `RMM_LOG_FILE` environment variable.
   * @param autoFlush If true, flushes the log for every (de)allocation. Warning, this will degrade
   *   performance.
   */
  constructor(upstreamMemoryResource: MemoryResource, logFilePath?: string, autoFlush?: boolean) {
    super(MemoryResourceType.LOGGING, upstreamMemoryResource, logFilePath, autoFlush);
  }
}
