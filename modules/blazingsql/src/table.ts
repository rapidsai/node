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

import {DataFrame, DataType} from '@rapidsai/cudf';
import {parseSchema} from './addon';

export interface ParsedSchema {
  files: string[];
  fileType: number;
  types: DataType[];
  names: string[];
  calciteToFileIndicies: number[];
  hasHeaderCSV: boolean;
}

export class SQLTable {
  public tableName: string;
  public tableSource: TableSource;

  constructor(tableName: string, input: DataFrame|string[]) {
    this.tableName   = tableName;
    this.tableSource = input instanceof DataFrame ? new DataFrameTable(input) : new CSVTable(input);
  }

  names(): string[] { return this.tableSource.names(); }

  type(columnName: string): DataType { return this.tableSource.type(columnName); }
}

interface TableSource {
  names(): string[];
  type(columnName: string): DataType;
  getSource(): any;
}

class CSVTable implements TableSource {
  private schema: ParsedSchema;

  constructor(input: string[]) { this.schema = parseSchema(input); }

  getSource() { return this.schema; }

  names(): string[] { return this.schema.names; }

  type(columnName: string): DataType {
    const idx = this.schema.names.indexOf(columnName);
    return this.schema.types[idx];
  }
}

export class DataFrameTable implements TableSource {
  private df: DataFrame;

  constructor(input: DataFrame) { this.df = input; }

  getSource() { return this.df; }

  names(): string[] { return this.df.names.concat(); }

  type(columnName: string): DataType { return this.df.get(columnName).type; }
}
