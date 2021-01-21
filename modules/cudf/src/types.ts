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

import * as ArrowDataType from 'apache-arrow/type';

import {Column} from './column';

/**
 * 		The desired order of null compared to other elements for a column.
 */
export enum NullOrder
{
  AFTER,
  BEFORE
}

export interface DataType<T extends TypeId = any> {
  readonly id: T;
  readonly valueType: any;
}
export class DataType<T extends TypeId = any> {
  constructor(public readonly id: T) {}
}

export type TypeMap = {
  [key: string]: DataType
}

export type ColumnsMap<T extends TypeMap> = {
  [P in keyof T]: Column<T[P]>
};

export enum TypeId
{
  EMPTY,
  INT8,
  INT16,
  INT32,
  INT64,
  UINT8,
  UINT16,
  UINT32,
  UINT64,
  FLOAT32,
  FLOAT64,
  BOOL8,
  TIMESTAMP_DAYS,
  TIMESTAMP_SECONDS,
  TIMESTAMP_MILLISECONDS,
  TIMESTAMP_MICROSECONDS,
  TIMESTAMP_NANOSECONDS,
  DURATION_DAYS,
  DURATION_SECONDS,
  DURATION_MILLISECONDS,
  DURATION_MICROSECONDS,
  DURATION_NANOSECONDS,
  DICTIONARY32,
  STRING,
  LIST,
  DECIMAL32,
  DECIMAL64,
}

export interface Int8 extends DataType<TypeId.INT8> {
  valueType: number;
}
export class Int8 extends DataType<TypeId.INT8> {
  constructor() { super(TypeId.INT8); }
}

export interface Int16 extends DataType<TypeId.INT16> {
  valueType: number;
}
export class Int16 extends DataType<TypeId.INT16> {
  constructor() { super(TypeId.INT16); }
}

export interface Int32 extends DataType<TypeId.INT32> {
  valueType: number;
}
export class Int32 extends DataType<TypeId.INT32> {
  constructor() { super(TypeId.INT32); }
}

export interface Int64 extends DataType<TypeId.INT64> {
  valueType: bigint;
}
export class Int64 extends DataType<TypeId.INT64> {
  constructor() { super(TypeId.INT64); }
}

export interface Uint8 extends DataType<TypeId.UINT8> {
  valueType: number;
}
export class Uint8 extends DataType<TypeId.UINT8> {
  constructor() { super(TypeId.UINT8); }
}

export interface Uint16 extends DataType<TypeId.UINT16> {
  valueType: number;
}
export class Uint16 extends DataType<TypeId.UINT16> {
  constructor() { super(TypeId.UINT16); }
}

export interface Uint32 extends DataType<TypeId.UINT32> {
  valueType: number;
}
export class Uint32 extends DataType<TypeId.UINT32> {
  constructor() { super(TypeId.UINT32); }
}

export interface Uint64 extends DataType<TypeId.UINT64> {
  valueType: bigint;
}
export class Uint64 extends DataType<TypeId.UINT64> {
  constructor() { super(TypeId.UINT64); }
}

export interface Float32 extends DataType<TypeId.FLOAT32> {
  valueType: number;
}
export class Float32 extends DataType<TypeId.FLOAT32> {
  constructor() { super(TypeId.FLOAT32); }
}

export interface Float64 extends DataType<TypeId.FLOAT64> {
  valueType: number;
}
export class Float64 extends DataType<TypeId.FLOAT64> {
  constructor() { super(TypeId.FLOAT64); }
}

export interface Bool8 extends DataType<TypeId.BOOL8> {
  valueType: boolean;
}
export class Bool8 extends DataType<TypeId.BOOL8> {
  constructor() { super(TypeId.BOOL8); }
}

export interface Utf8String extends DataType<TypeId.STRING> {
  valueType: string;
}
export class Utf8String extends DataType<TypeId.STRING> {
  constructor() { super(TypeId.STRING); }
}

export type CUDFToArrowType<T extends DataType> = {
  [TypeId.INT8]: ArrowDataType.Int8,
  [TypeId.INT16]: ArrowDataType.Int16,
  [TypeId.INT32]: ArrowDataType.Int32,
  [TypeId.INT64]: ArrowDataType.Int64,
  [TypeId.UINT8]: ArrowDataType.Uint8,
  [TypeId.UINT16]: ArrowDataType.Uint16,
  [TypeId.UINT32]: ArrowDataType.Uint32,
  [TypeId.UINT64]: ArrowDataType.Uint64,
  [TypeId.FLOAT32]: ArrowDataType.Float32,
  [TypeId.FLOAT64]: ArrowDataType.Float64,
  [TypeId.BOOL8]: ArrowDataType.Bool,
  [TypeId.STRING]: ArrowDataType.Utf8,
}[T['id']];

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

interface ReadCSVOptionsBase<T extends CSVTypeMap = any> {
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

interface ReadCSVFileOptions<T extends CSVTypeMap = any> extends ReadCSVOptionsBase<T> {
  sourceType: "files";
  sources: string[];
}

interface ReadCSVBufferOptions<T extends CSVTypeMap = any> extends ReadCSVOptionsBase<T> {
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

export type Integral = Int8|Int16|Int32|Uint8|Uint16|Uint32;
