// Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

import {setDefaultAllocator} from '@rapidsai/cuda';
import {Series} from '@rapidsai/cudf';
import {renumberNodes} from '@rapidsai/cugraph';
import {CudaMemoryResource, DeviceBuffer} from '@rapidsai/rmm';

const mr = new CudaMemoryResource();

setDefaultAllocator((byteLength: number) => new DeviceBuffer(byteLength, mr));

test('renumberNodes strings', () => {
  const src = Series.new([
    '192.168.1.1',
    '172.217.5.238',
    '216.228.121.209',
    '192.16.31.23',
  ]);
  const dst = Series.new([
    '172.217.5.238',
    '216.228.121.209',
    '192.16.31.23',
    '192.168.1.1',
  ]);
  const df  = renumberNodes(src, dst);
  expect([...df.get('node')].sort()).toEqual([
    '192.168.1.1',
    '172.217.5.238',
    '216.228.121.209',
    '192.16.31.23',
  ].sort());
  expect([...df.get('id')]).toEqual([0, 1, 2, 3]);
});

test('renumberNodes numeric', () => {
  const src = Series.new([
    10,
    20,
    30,
    40,
  ]);
  const dst = Series.new([
    20,
    30,
    40,
    10,
  ]);
  const df  = renumberNodes(src, dst);
  expect([...df.get('node')].sort()).toEqual([
    10,
    20,
    30,
    40,
  ].sort());
  expect([...df.get('id')]).toEqual([0, 1, 2, 3]);
});
