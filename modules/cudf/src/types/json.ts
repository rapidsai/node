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

import {TypeMap} from './mappings';

export interface ReadJSONOptionsCommon<T extends TypeMap = any> {
  /**
     Names and types of all the fields; if empty then names and types are inferred/auto-generated
   */
  dataTypes?: T;
  /** The compression format of the source, or infer from file extension */
  compression?: 'infer'|'snappy'|'gzip'|'bz2'|'brotli'|'zip'|'xz';
  /** The number of bytes to skip from source start */
  byteOffset?: number;
  /** The number of bytes to read */
  byteRange?: number;
  /** Set whether to read the file as a json object per line (required to be true) */
  lines?: boolean;
  /** Set wehther to parse dates as DD/MM versus MM/DD*/
  dayfirst?: boolean;
}

export interface ReadJSONFileOptions<T extends TypeMap = any> extends ReadJSONOptionsCommon<T> {
  sourceType: 'files';
  sources: string[];
}

export interface ReadJSONBufferOptions<T extends TypeMap = any> extends ReadJSONOptionsCommon<T> {
  sourceType: 'buffers';
  sources: (Uint8Array|Buffer)[];
}

export type ReadJSONOptions<T extends TypeMap = any> =
  ReadJSONFileOptions<T>|ReadJSONBufferOptions<T>;
