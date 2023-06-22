// Copyright (c) 2023, NVIDIA CORPORATION.
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

import regl from 'regl';

export class ReglRingBuffer {
  private buffer: regl.Buffer;
  private head: number;
  private bufferSize: number;

  constructor(instance: regl.Regl, size: number) {
    this.bufferSize = size;
    this.buffer     = instance.buffer({
      usage: 'dynamic',
      type: 'float',
      length: size * 4 * 2,
    });
    this.head       = 0;
  }

  write(newData: Float32Array): void {
    const size = newData.length;
    if (size > this.bufferSize) {
      throw new Error(
        `Single write of ${size} elements cannot exceed RingBuffer size ${this.bufferSize}`);
    }
    if (this.head === this.bufferSize) { this.head = 0; }

    if (this.head + size > this.bufferSize) {
      const left = this.bufferSize - this.head;
      this.buffer.subdata(newData.slice(0, left), this.head);
      this.buffer.subdata(newData.slice(left), 0);
      this.head = (this.head + size) % this.bufferSize;
    } else {
      this.buffer.subdata(newData, this.head);
      this.head += size;
    }
  }

  get(): regl.Buffer { return this.buffer; }

  destroy(): void { this.buffer.destroy(); }
}
