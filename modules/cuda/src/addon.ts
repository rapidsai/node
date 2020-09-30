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

export const CUDA = (() => {
    let CUDA: any, types = ['Release'];
    if (process.env.NODE_DEBUG !== undefined || process.env.NODE_ENV === 'debug') {
        types.push('Debug');
    }
    for (let type; type = types.pop();) {
        try {
            if (CUDA = require(`../${type}/node_cuda.node`)) {
                break;
            }
        } catch (e) { console.error(e); continue; }
    }
    if (CUDA) return CUDA.init();
    throw new Error('node_cuda not found');
})();

export default CUDA;

export interface CUDA {
    CUDAArray: CUDAArrayConstructor;
    CUDABuffer: CUDABufferConstructor;
}

export interface CUDAArrayConstructor {
    readonly prototype: CUDAArray;
    new(): CUDAArray;
}

export interface CUDAArray {
    readonly ary: number;
    readonly byteLength: number;
    readonly bytesPerElement: number;
    readonly width: number;
    readonly height: number;
    readonly depth: number;
    readonly channelFormatX: number;
    readonly channelFormatY: number;
    readonly channelFormatZ: number;
    readonly channelFormatW: number;
    readonly channelFormatKind: number;
}

export declare var CUDAArray: CUDAArrayConstructor;

export interface CUDABufferConstructor {
    readonly prototype: CUDABuffer;
    new(): CUDABuffer;
}

export interface CUDABuffer {
    readonly ptr: number;
    readonly byteLength: number;
    slice(begin: number, end?: number): CUDABuffer;
}

export declare var CUDABuffer: CUDABufferConstructor;
