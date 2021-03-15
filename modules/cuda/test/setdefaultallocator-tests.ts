// Copyright (c) 2021, NVIDIA CORPORATION.
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

import {test} from '@jest/globals';
import {
  DeviceMemory,
  Float32Buffer,
  ManagedMemory,
  PinnedMemory,
  setDefaultAllocator
} from '@nvidia/cuda';

test('setDefaultAllocator with null resets to the default DeviceMemory allocator', () => {
  // Set a custom allocate fn
  setDefaultAllocator((n) => new ManagedMemory(n));
  const mbuf = new Float32Buffer(1024).fill(100);
  expect(mbuf.buffer).toBeInstanceOf(ManagedMemory);
  // Reset to the default
  setDefaultAllocator(null);
  const dbuf = new Float32Buffer(1024).fill(100);
  expect(dbuf.buffer).toBeInstanceOf(DeviceMemory);
});

test.each([
  DeviceMemory,
  PinnedMemory,
  ManagedMemory,
])(`setDefaultAllocator works with %s`, (MemoryCtor) => {
  setDefaultAllocator((n) => new MemoryCtor(n));
  const buf = new Float32Buffer(1024).fill(100);
  expect(buf.buffer).toBeInstanceOf(MemoryCtor);
  expect(buf.toArray()).toEqual(new Float32Array(1024).fill(100));
  setDefaultAllocator(null);
});
