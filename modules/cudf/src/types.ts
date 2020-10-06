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


export enum types {
    EMPTY = CUDF.types.typeID.EMPTY,
    INT8 = CUDF.types.typeID.INT8,
    INT16 = CUDF.types.typeID.INT16,
    INT32 = CUDF.types.typeID.INT32,
    INT64 = CUDF.types.typeID.INT64,
    UINT8 = CUDF.types.typeID.UINT8,
    UINT16 = CUDF.types.typeID.UINT16,
    UINT32 = CUDF.types.typeID.UINT32,
    UINT64 = CUDF.types.typeID.UINT64,
    FLOAT32 = CUDF.types.typeID.FLOAT32,
    FLOAT64 = CUDF.types.typeID.FLOAT64,
    BOOL8 = CUDF.types.typeID.BOOL8,
    TIMESTAMP_DAYS = CUDF.types.typeID.TIMESTAMP_DAYS,
    TIMESTAMP_SECONDS = CUDF.types.typeID.TIMESTAMP_SECONDS,
    TIMESTAMP_MILLISECONDS = CUDF.types.typeID.TIMESTAMP_MILLISECONDS,
    TIMESTAMP_MICROSECONDS = CUDF.types.typeID.TIMESTAMP_MICROSECONDS,
    TIMESTAMP_NANOSECONDS = CUDF.types.typeID.TIMESTAMP_NANOSECONDS,
    DURATION_DAYS = CUDF.types.typeID.DURATION_DAYS,
    DURATION_SECONDS = CUDF.types.typeID.DURATION_SECONDS,
    DURATION_MILLISECONDS = CUDF.types.typeID.DURATION_MILLISECONDS,
    DURATION_NANOSECONDS = CUDF.types.typeID.DURATION_NANOSECONDS,
    DICTIONARY32 = CUDF.types.typeID.DICTIONARY32,
    STRING = CUDF.types.typeID.STRING,
    LIST = CUDF.types.typeID.LIST,
    DECIMAL32 = CUDF.types.typeID.DECIMAL32,
    DECIMAL64 = CUDF.types.typeID.DECIMAL64,
}
