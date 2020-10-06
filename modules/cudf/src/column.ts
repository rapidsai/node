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

import CUDF from './addon';
import { types } from './types';
import { DeviceBuffer } from '@nvidia/rmm';

export interface ColumnConstructor {
    readonly prototype: Column;
    new(
        dtype?: types, size?:number, data?: DeviceBuffer,
        null_mask?: DeviceBuffer, null_count?: number
    ): Column;
}

export interface Column {
    type(): types;
    size(): number;
    null_count(): number;
    set_null_count(count_:number): void;
    set_null_mask(new_null_mask:DeviceBuffer, new_null_count?:number): void;
    nullable(): boolean;
    has_nulls(): boolean;
}

export const Column: ColumnConstructor = CUDF.Column;
