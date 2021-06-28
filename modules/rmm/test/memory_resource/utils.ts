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

import {
  BinningMemoryResource,
  CudaMemoryResource,
  FixedSizeMemoryResource,
  LoggingResourceAdapter,
  ManagedMemoryResource,
  MemoryResource,
  PoolMemoryResource,
} from '@rapidsai/rmm';
import {mkdtempSync} from 'fs';
import * as Path from 'path';

import {sizes} from '../utils';

type TestConfig = {
  comparable: boolean; supportsStreams: boolean; supportsGetMemInfo: boolean;
  createMemoryResource(): MemoryResource;
};

let logFileDir = '', logFilePath = '';

beforeAll(() => {
  logFileDir  = mkdtempSync(Path.join('/tmp', 'node_rmm'));
  logFilePath = Path.join(logFileDir, 'log');
});

afterAll(() => {
  const rimraf = require('rimraf');
  return new Promise<void>((resolve, reject) => {  //
    rimraf(logFileDir, (err?: Error|null) => err ? reject(err) : resolve());
  });
});

export const memoryResourceTestConfigs = [
  [
    `CudaMemoryResource`,
    {
      comparable: true,
      supportsStreams: false,
      supportsGetMemInfo: true,
      createMemoryResource: () => new CudaMemoryResource(),
    }
  ],
  [
    `ManagedMemoryResource`,
    {
      comparable: true,
      supportsStreams: false,
      supportsGetMemInfo: true,
      createMemoryResource: () => new ManagedMemoryResource(),
    }
  ],
  [
    `PoolMemoryResource`,
    {
      comparable: false,
      supportsStreams: true,
      supportsGetMemInfo: false,
      createMemoryResource: () =>
        new PoolMemoryResource(new CudaMemoryResource(), sizes['1_MiB'], sizes['16_MiB']),
    }
  ],
  [
    `FixedSizeMemoryResource`,
    {
      comparable: false,
      supportsStreams: true,
      supportsGetMemInfo: false,
      createMemoryResource: () =>
        new FixedSizeMemoryResource(new CudaMemoryResource(), sizes['4_MiB'], 1),
    }
  ],
  [
    `BinningMemoryResource`,
    {
      comparable: false,
      supportsStreams: true,
      supportsGetMemInfo: false,
      createMemoryResource: () => new BinningMemoryResource(
        new CudaMemoryResource(), Math.log2(sizes['1_MiB']), Math.log2(sizes['1_MiB'])),
    }
  ],
  [
    `LoggingResourceAdapter`,
    {
      comparable: true,
      supportsStreams: false,
      supportsGetMemInfo: true,
      createMemoryResource: () =>
        new LoggingResourceAdapter(new CudaMemoryResource(), logFilePath, true),
    }
  ],
] as [string, TestConfig][];
