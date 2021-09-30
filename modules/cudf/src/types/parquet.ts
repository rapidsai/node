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

export interface ReadParquetOptions {
  sources: string[];
  /** The list of columns to read */
  columns?: string[];
  /** The number of rows to skip from the start of the file */
  skipRows?: number;
  /** The total number of rows to read */
  numRows?: number;
  /** Return string columns as GDF_CATEGORY dtype */
  stringsToCategorical?: boolean;
  /**
   * If true and dataset has custom PANDAS schema metadata, ensure that index columns are also
   * loaded.
   */
  usePandasMetadata?: boolean;
}
