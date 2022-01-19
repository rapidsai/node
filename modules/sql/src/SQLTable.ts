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

import {arrowToCUDFType, DataFrame, DataType} from '@rapidsai/cudf';
import {parseSchema} from './addon';

export interface ParsedSchema {
  files: string[];
  fileType: number;
  types: DataType[];
  names: string[];
  calciteToFileIndicies: number[];
  hasHeaderCSV: boolean;
}

export interface SQLTable {
  tableName: string;
  get names(): string[];
  type(columnName: string): DataType;
  getSource(): any;
}

export class FileTable implements SQLTable {
  public tableName: string;
  private schema: ParsedSchema;

  constructor(tableName: string, input: string[], fileType: 'csv'|'orc'|'parquet') {
    this.tableName = tableName;
    this.schema    = parseSchema(input, fileType);
  }

  get names(): string[] { return this.schema.names; }
  getSource() { return this.schema; }
  type(columnName: string): DataType {
    const idx = this.schema.names.indexOf(columnName);
    return arrowToCUDFType(this.schema.types[idx]);
  }
}

export class DataFrameTable implements SQLTable {
  public tableName: string;
  private df: DataFrame;

  constructor(tableName: string, input: DataFrame) {
    this.tableName = tableName;
    this.df        = input;
  }

  get names(): string[] { return this.df.names.concat(); }
  getSource() { return this.df; }
  type(columnName: string): DataType { return this.df.get(columnName).type; }
}
