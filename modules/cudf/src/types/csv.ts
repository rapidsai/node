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

import {
  Bool8,
  Float32,
  Float64,
  Int16,
  Int32,
  Int64,
  Int8,
  Uint16,
  Uint32,
  Uint64,
  Uint8,
  Utf8String
} from './dtypes';

export type CSVType = "int8"|"int16"|"int32"|"int64"|"uint8"|"uint16"|"uint32"|"uint64"|"float32"|
  "float64"|"datetime64[s]"|"datetime64[ms]"|"datetime64[us]"|"datetime64[ns]"|"timedelta64[s]"|
  "timedelta64[ms]"|"timedelta64[us]"|"timedelta64[ns]"|"bool"|"category"|"str"|"hex"|"hex32"|
  "hex64"|"date"|"date32"|"date64"|"timestamp"|"timestamp[us]"|"timestamp[s]"|"timestamp[ms]"|
  "timestamp[ns]";

export type CSVTypeMap = {
  [key: string]: CSVType;
};

export type CSVToCUDFType<T extends CSVType> = {
  "int8": Int8,
  "int16": Int16,
  "int32": Int32,
  "int64": Int64,
  "uint8": Uint8,
  "uint16": Uint16,
  "uint32": Uint32,
  "uint64": Uint64,
  "float32": Float32,
  "float64": Float64,
  "datetime64[s]": never,
  "datetime64[ms]": never,
  "datetime64[us]": never,
  "datetime64[ns]": never,
  "timedelta64[s]": never,
  "timedelta64[ms]": never,
  "timedelta64[us]": never,
  "timedelta64[ns]": never,
  "bool": Bool8,
  "category": never,
  "str": Utf8String,
  "hex": never,
  "hex32": never,
  "hex64": never,
  "date": never,
  "date32": never,
  "date64": never,
  "timestamp": never,
  "timestamp[us]": never,
  "timestamp[s]": never,
  "timestamp[ms]": never,
  "timestamp[ns]": never,
}[T];

export interface ReadCSVOptionsCommon<T extends CSVTypeMap = any> {
  /**
     Names and types of all the columns; if empty then names and types are inferred/auto-generated
   */
  dataTypes?: T;
  /** The compression format of the source, or infer from file extension */
  compression?: "infer"|"snappy"|"gzip"|"bz2"|"brotli"|"zip"|"xz";
  /** Whether to rename duplicate column names */
  renameDuplicateColumns?: boolean;
  /** Rows to read; -1 is all */
  numRows?: number;
  /** Rows to skip from the start */
  skipHead?: number;
  /** Rows to skip from the end */
  skipTail?: number;
  /** Treatment of quoting behavior */
  quoteStyle?: "all"|"none"|"nonnumeric"|"minimal";
  /** Line terminator */
  lineTerminator?: string;
  /** Quoting character (if `allowDoubleQuoting` is true) */
  quoteCharacter?: string;
  /** Decimal point character; cannot match delimiter */
  decimalCharacter?: string;
  /** Treat whitespace as field delimiter; overrides character delimiter */
  whitespaceAsDelimiter?: boolean;
  /** Whether to skip whitespace after the delimiter */
  skipInitialSpaces?: boolean;
  /** Ignore empty lines or parse line values as invalid */
  skipBlankLines?: boolean;
  /** Whether a quote inside a value is double-quoted */
  allowDoubleQuoting?: boolean;
  /** Whether to keep the built-in default NA values */
  keepDefaultNA?: boolean;
  /** Whether to disable null filter; disabling can improve performance */
  autoDetectNullValues?: boolean;
  /** Whether to parse dates as DD/MM versus MM/DD */
  inferDatesWithDayFirst?: boolean;
  /** Field delimiter */
  delimiter?: string;
  /** Numeric data thousands separator; cannot match delimiter */
  thousands?: string;
  /** Comment line start character */
  comment?: string;
  /** Header row index */
  header?: "infer"|null|number;
  /** String used as prefix for each column name if no header or names are provided. */
  prefix?: string;
  /** Additional values to recognize as null values */
  nullValues?: string[];
  /** Additional values to recognize as boolean true values */
  trueValues?: string[];
  /** Additional values to recognize as boolean false values */
  falseValues?: string[];
  /** Names of columns to read as datetime */
  datetimeColumns?: string[];
  /** Names of columns to read; empty/null is all columns */
  columnsToReturn?: string[];
}

export interface ReadCSVFileOptions<T extends CSVTypeMap = any> extends ReadCSVOptionsCommon<T> {
  sourceType: "files";
  sources: string[];
}

export interface ReadCSVBufferOptions<T extends CSVTypeMap = any> extends ReadCSVOptionsCommon<T> {
  sourceType: "buffers";
  sources: (Uint8Array|Buffer)[];
}

export type ReadCSVOptions<T extends CSVTypeMap = any> =
  ReadCSVFileOptions<T>|ReadCSVBufferOptions<T>;

export interface WriteCSVOptions {
  /** The field delimiter to write. */
  delimiter?: string;  // = ",";
  /** String to use for null values. */
  nullValue?: string;
  /** String to use for boolean true values (default 'true'). */
  trueValue?: string;
  /** String to use for boolean false values (default 'false'). */
  falseValue?: string;
  /** Indicates whether to write headers to csv. */
  includeHeader?: boolean;
  /** Character to use for separating lines, */
  lineTerminator?: string;
  /** Maximum number of rows to write in each chunk (limits memory use). */
  rowsPerChunk?: number;
}
