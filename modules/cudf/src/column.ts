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
import { DeviceBuffer } from '@nvidia/rmm';

// const data_types = [
// "empty",
// "int8", 
// "int16",
// "int32",
// "int64",
// "uint8",
// "uint16",
// "uint32",
// "uint64",
// "float32",
// "float64",
// "bool8",
// "timestamp_days",
// "timestamp_seconds",
// "timestamp_milliseconds",
// "timestamp_microseconds",
// "timestamp_nanoseconds",
// "duration_days",
// "duration_seconds",
// "duration_milliseconds",
// "duration_microseconds",
// "duration_nanoseconds", 
// "dictionary32",
// "string",
// "list",
// "decimal32",
// "decimal64",
// ]

export interface ColumnConstructor {
    readonly prototype: Column;
    new(
        dtype?: string, size?:number, data?: DeviceBuffer,
        null_mask?: DeviceBuffer, null_count?: number
    ): Column;
}

export interface Column {
    type(): string;
    size(): number;
    null_count(): number;
    set_null_count(count_:number): void;
    nullable(): boolean;
    has_nulls(): boolean;
    release(): void;
}

export const Column: ColumnConstructor = CUDF.Column;
