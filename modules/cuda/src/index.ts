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

import * as CUDA from './cuda';
import { CUDADevice } from './device';

export { CUDA };
export * from './array';
export { CUDADevice } from './device';
export { CUDAMemory } from './memory';
export { CUDADeviceFlag } from './cuda';
export { CUDAMemHostAllocFlag } from './cuda';
export { CUDAMemHostRegisterFlag } from './cuda';
export { CUDAGraphicsRegisterFlag } from './cuda';

interface CUDADeviceList extends Iterable<CUDADevice> {
    length: number; [key: number]: CUDADevice;
};

export const devices = new Proxy<CUDADeviceList>({
    length: CUDA.device.getCount(),
    *[Symbol.iterator]() {
        for (let i = -1, n = this.length; ++i < n; yield this[i]);
    }
}, {
        isExtensible() { return false; },
        set() { throw new Error('Invalid operation'); },
        defineProperty() { throw new Error('Invalid operation'); },
        deleteProperty() { throw new Error('Invalid operation'); },
        has(target, key) {
            return typeof key !== 'number' ? key in target : key > -1 && key < target.length;
        },
        get(target, key) {
            let idx = typeof key !== 'symbol' ? +(key as any) : NaN;
            if (idx == idx && idx > -1 && idx < target.length) {
                return target[idx] ? target[idx].activate() : (target[idx] = CUDADevice.new(idx));
            }
            return target[key as any];
        },
    }
);
