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

export interface DataTypeConstructor {
  new(id: TypeId): DataType;
}

export interface DataType {
  readonly id: TypeId;
}

// eslint-disable-next-line @typescript-eslint/no-redeclare
export const DataType: DataTypeConstructor = CUDF.DataType;

export enum TypeId
{
  EMPTY                  = CUDF.TypeId.EMPTY,
  INT8                   = CUDF.TypeId.INT8,
  INT16                  = CUDF.TypeId.INT16,
  INT32                  = CUDF.TypeId.INT32,
  INT64                  = CUDF.TypeId.INT64,
  UINT8                  = CUDF.TypeId.UINT8,
  UINT16                 = CUDF.TypeId.UINT16,
  UINT32                 = CUDF.TypeId.UINT32,
  UINT64                 = CUDF.TypeId.UINT64,
  FLOAT32                = CUDF.TypeId.FLOAT32,
  FLOAT64                = CUDF.TypeId.FLOAT64,
  BOOL8                  = CUDF.TypeId.BOOL8,
  TIMESTAMP_DAYS         = CUDF.TypeId.TIMESTAMP_DAYS,
  TIMESTAMP_SECONDS      = CUDF.TypeId.TIMESTAMP_SECONDS,
  TIMESTAMP_MILLISECONDS = CUDF.TypeId.TIMESTAMP_MILLISECONDS,
  TIMESTAMP_MICROSECONDS = CUDF.TypeId.TIMESTAMP_MICROSECONDS,
  TIMESTAMP_NANOSECONDS  = CUDF.TypeId.TIMESTAMP_NANOSECONDS,
  DURATION_DAYS          = CUDF.TypeId.DURATION_DAYS,
  DURATION_SECONDS       = CUDF.TypeId.DURATION_SECONDS,
  DURATION_MILLISECONDS  = CUDF.TypeId.DURATION_MILLISECONDS,
  DURATION_NANOSECONDS   = CUDF.TypeId.DURATION_NANOSECONDS,
  DICTIONARY32           = CUDF.TypeId.DICTIONARY32,
  STRING                 = CUDF.TypeId.STRING,
  LIST                   = CUDF.TypeId.LIST,
  DECIMAL32              = CUDF.TypeId.DECIMAL32,
  DECIMAL64              = CUDF.TypeId.DECIMAL64,
}
