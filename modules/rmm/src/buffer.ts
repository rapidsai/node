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

export interface DeviceBufferConstructor {
    readonly prototype: DeviceBuffer;
    new(byteLength?: number, stream?: number): DeviceBuffer;
}

export interface DeviceBuffer {
    readonly ptr: number;
    readonly byteLength: number;
    slice(begin: number, end?: number): DeviceBuffer;
}

export const DeviceBuffer: DeviceBufferConstructor = RMM.DeviceBuffer;
