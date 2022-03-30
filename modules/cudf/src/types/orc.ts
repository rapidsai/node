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

export interface ReadORCOptionsCommon {
  /** The list of columns to read */
  columns?: string[];
  /** Only these stripes will be read from the file. */
  stripes?: number[][];
  /** The number of rows to skip from the start of the file */
  skipRows?: number;
  /** The total number of rows to read */
  numRows?: number;
  /** Use row index if available for faster seeking (default 'true') */
  useIndex?: boolean;
  /** Names of the columns that should be read as 128-bit Decimal */
  decimalColumns?: string[];
}

export interface ReadORCFileOptions extends ReadORCOptionsCommon {
  sourceType: 'files';
  sources: string[];
}

export interface ReadORCBufferOptions extends ReadORCOptionsCommon {
  sourceType: 'buffers';
  sources: (Uint8Array|Buffer)[];
}

export type ReadORCOptions = ReadORCFileOptions|ReadORCBufferOptions;

export interface WriteORCOptions {
  /** The name of compression to use (default 'None'). */
  compression?: 'snappy'|'none';
  /** Write timestamps in int96 format (default 'true'). */
  enableStatistics?: boolean;
}

export interface TableWriteORCOptions {
  /** Column names to write in the header. */
  columnNames?: string[];
}
