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

import * as arrowEnums from 'apache-arrow/enum';
import * as arrowTypes from 'apache-arrow/type';

import CUDF from './addon';
import {Column} from './column';
import {Bool8Series} from './series/bool';
import {Float32Series, Float64Series} from './series/float';
import {
  Int16Series,
  Int32Series,
  Int64Series,
  Int8Series,
  Uint16Series,
  Uint32Series,
  Uint64Series,
  Uint8Series
} from './series/integral';
import {StringSeries} from './series/string';

/**
 * The desired order of null compared to other elements for a column.
 */
export enum NullOrder
{
  AFTER,
  BEFORE
}

interface DataTypeConstructor {
  readonly prototype: DataType;
  new<T extends TypeId = any>(id: T): DataType<T>;
}

export interface DataType<T extends TypeId = any> {
  readonly id: T;
  readonly valueType: any;
  readonly BYTES_PER_ELEMENT: number;
}

// eslint-disable-next-line @typescript-eslint/no-redeclare
export const DataType: DataTypeConstructor = CUDF.DataType;

(DataType.prototype as any).BYTES_PER_ELEMENT = 0;

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
  readonly BYTES_PER_ELEMENT: number = 1;
}

export interface Int16 extends DataType<TypeId.INT16> {
  valueType: number;
}
export class Int16 extends DataType<TypeId.INT16> {
  constructor() { super(TypeId.INT16); }
  readonly BYTES_PER_ELEMENT: number = 2;
}

export interface Int32 extends DataType<TypeId.INT32> {
  valueType: number;
}
export class Int32 extends DataType<TypeId.INT32> {
  constructor() { super(TypeId.INT32); }
  readonly BYTES_PER_ELEMENT: number = 4;
}

export interface Int64 extends DataType<TypeId.INT64> {
  valueType: bigint;
}
export class Int64 extends DataType<TypeId.INT64> {
  constructor() { super(TypeId.INT64); }
  readonly BYTES_PER_ELEMENT: number = 8;
}

export interface Uint8 extends DataType<TypeId.UINT8> {
  valueType: number;
}
export class Uint8 extends DataType<TypeId.UINT8> {
  constructor() { super(TypeId.UINT8); }
  readonly BYTES_PER_ELEMENT: number = 1;
}

export interface Uint16 extends DataType<TypeId.UINT16> {
  valueType: number;
}
export class Uint16 extends DataType<TypeId.UINT16> {
  constructor() { super(TypeId.UINT16); }
  readonly BYTES_PER_ELEMENT: number = 2;
}

export interface Uint32 extends DataType<TypeId.UINT32> {
  valueType: number;
}
export class Uint32 extends DataType<TypeId.UINT32> {
  constructor() { super(TypeId.UINT32); }
  readonly BYTES_PER_ELEMENT: number = 4;
}

export interface Uint64 extends DataType<TypeId.UINT64> {
  valueType: bigint;
}
export class Uint64 extends DataType<TypeId.UINT64> {
  constructor() { super(TypeId.UINT64); }
  readonly BYTES_PER_ELEMENT: number = 8;
}

export interface Float32 extends DataType<TypeId.FLOAT32> {
  valueType: number;
}
export class Float32 extends DataType<TypeId.FLOAT32> {
  constructor() { super(TypeId.FLOAT32); }
  readonly BYTES_PER_ELEMENT: number = 4;
}

export interface Float64 extends DataType<TypeId.FLOAT64> {
  valueType: number;
}
export class Float64 extends DataType<TypeId.FLOAT64> {
  constructor() { super(TypeId.FLOAT64); }
  readonly BYTES_PER_ELEMENT: number = 8;
}

export interface Bool8 extends DataType<TypeId.BOOL8> {
  valueType: boolean;
}
export class Bool8 extends DataType<TypeId.BOOL8> {
  constructor() { super(TypeId.BOOL8); }
  readonly BYTES_PER_ELEMENT: number = 1;
}

export interface Utf8String extends DataType<TypeId.STRING> {
  valueType: string;
}
export class Utf8String extends DataType<TypeId.STRING> {
  constructor() { super(TypeId.STRING); }
}

export type ArrowToCUDFType<T extends arrowTypes.DataType> = {
  [arrowEnums.Type.NONE]: never,
  [arrowEnums.Type.Null]: never,
  [arrowEnums.Type.Int]: never,
  [arrowEnums.Type.Float]: never,
  [arrowEnums.Type.Binary]: never,
  [arrowEnums.Type.Bool]: Bool8,
  [arrowEnums.Type.Utf8]: Utf8String,
  [arrowEnums.Type.Decimal]: never,
  [arrowEnums.Type.Date]: never,
  [arrowEnums.Type.Time]: never,
  [arrowEnums.Type.Timestamp]: never,
  [arrowEnums.Type.Interval]: never,
  [arrowEnums.Type.List]: never,
  [arrowEnums.Type.Struct]: never,
  [arrowEnums.Type.Union]: never,
  [arrowEnums.Type.FixedSizeBinary]: never,
  [arrowEnums.Type.FixedSizeList]: never,
  [arrowEnums.Type.Map]: never,
  [arrowEnums.Type.Dictionary]: never,
  [arrowEnums.Type.Int8]: Int8,
  [arrowEnums.Type.Int16]: Int16,
  [arrowEnums.Type.Int32]: Int32,
  [arrowEnums.Type.Int64]: Int64,
  [arrowEnums.Type.Uint8]: Uint8,
  [arrowEnums.Type.Uint16]: Uint16,
  [arrowEnums.Type.Uint32]: Uint32,
  [arrowEnums.Type.Uint64]: Uint64,
  [arrowEnums.Type.Float16]: never,
  [arrowEnums.Type.Float32]: Float32,
  [arrowEnums.Type.Float64]: Float64,
  [arrowEnums.Type.DateDay]: never,
  [arrowEnums.Type.DateMillisecond]: never,
  [arrowEnums.Type.TimestampSecond]: never,
  [arrowEnums.Type.TimestampMillisecond]: never,
  [arrowEnums.Type.TimestampMicrosecond]: never,
  [arrowEnums.Type.TimestampNanosecond]: never,
  [arrowEnums.Type.TimeSecond]: never,
  [arrowEnums.Type.TimeMillisecond]: never,
  [arrowEnums.Type.TimeMicrosecond]: never,
  [arrowEnums.Type.TimeNanosecond]: never,
  [arrowEnums.Type.DenseUnion]: never,
  [arrowEnums.Type.SparseUnion]: never,
  [arrowEnums.Type.IntervalDayTime]: never,
  [arrowEnums.Type.IntervalYearMonth]: never,
}[T['typeId']];

export type CUDFToArrowType<T extends DataType> = {
  [TypeId.INT8]: arrowTypes.Int8,
  [TypeId.INT16]: arrowTypes.Int16,
  [TypeId.INT32]: arrowTypes.Int32,
  [TypeId.INT64]: arrowTypes.Int64,
  [TypeId.UINT8]: arrowTypes.Uint8,
  [TypeId.UINT16]: arrowTypes.Uint16,
  [TypeId.UINT32]: arrowTypes.Uint32,
  [TypeId.UINT64]: arrowTypes.Uint64,
  [TypeId.FLOAT32]: arrowTypes.Float32,
  [TypeId.FLOAT64]: arrowTypes.Float64,
  [TypeId.BOOL8]: arrowTypes.Bool,
  [TypeId.STRING]: arrowTypes.Utf8,
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

export type FloatingPoint = Float32|Float64;
export type Integral      = Int8|Int16|Int32|Uint8|Uint16|Uint32;
export type Numeric       = Integral|FloatingPoint|Int64|Uint64|Bool8;

export type TypeIdToType<T extends TypeId> = {
  [TypeId.EMPTY]: never,
  [TypeId.INT8]: Int8,
  [TypeId.INT16]: Int16,
  [TypeId.INT32]: Int32,
  [TypeId.INT64]: Int64,
  [TypeId.UINT8]: Uint8,
  [TypeId.UINT16]: Uint16,
  [TypeId.UINT32]: Uint32,
  [TypeId.UINT64]: Uint64,
  [TypeId.FLOAT32]: Float32,
  [TypeId.FLOAT64]: Float64,
  [TypeId.BOOL8]: Bool8,
  [TypeId.TIMESTAMP_DAYS]: never,
  [TypeId.TIMESTAMP_SECONDS]: never,
  [TypeId.TIMESTAMP_MILLISECONDS]: never,
  [TypeId.TIMESTAMP_MICROSECONDS]: never,
  [TypeId.TIMESTAMP_NANOSECONDS]: never,
  [TypeId.DURATION_DAYS]: never,
  [TypeId.DURATION_SECONDS]: never,
  [TypeId.DURATION_MILLISECONDS]: never,
  [TypeId.DURATION_MICROSECONDS]: never,
  [TypeId.DURATION_NANOSECONDS]: never,
  [TypeId.DICTIONARY32]: never,
  [TypeId.STRING]: Utf8String,
  [TypeId.LIST]: never,
  [TypeId.DECIMAL32]: never,
  [TypeId.DECIMAL64]: never,
}[T];

type CommonType_Bool8<T extends Numeric> = T;
type CommonType_Int8<T extends Numeric>  = T extends Bool8 ? Int8 : T;
type CommonType_Int16<T extends Numeric> = T extends Bool8|Int8|Uint8 ? Int16 : T;
type CommonType_Int32<T extends Numeric> = T extends Bool8|Int8|Uint8|Int16|Uint16 ? Int32 : T;
type CommonType_Int64<T extends Numeric> =
  T extends Bool8|Int8|Uint8|Int16|Uint16|Int32|Uint32 ? Int64 : T;
type CommonType_Uint8<T extends Numeric>  = T extends Bool8|Int8 ? Uint16 : T;
type CommonType_Uint16<T extends Numeric> = T extends Bool8|Int8|Uint8 ? Uint16 : T;
type CommonType_Uint32<T extends Numeric> = T extends Bool8|Int8|Uint8|Int16|Uint16 ? Uint32 : T;
type CommonType_Uint64<T extends Numeric> =
  T extends Bool8|Int8|Uint8|Int16|Uint16|Int32|Uint32|Int64 ? Uint64 : T;
type CommonType_Float32<T extends Numeric> =
  T extends Bool8|Int8|Uint8|Int16|Uint16 ? Float32 : Float64;
type CommonType_Float64<T extends Numeric> =
  T extends Bool8|Int8|Uint8|Int16|Uint16|Int32|Uint32|Float32 ? Float64 : T;

// clang-format off
export type CommonType<T extends Numeric, R extends Numeric> =
    T extends Bool8   ? CommonType_Bool8<R>
  : T extends Int8    ? CommonType_Int8<R>
  : T extends Int16   ? CommonType_Int16<R>
  : T extends Int32   ? CommonType_Int32<R>
  : T extends Int64   ? CommonType_Int64<R>
  : T extends Uint8   ? CommonType_Uint8<R>
  : T extends Uint16  ? CommonType_Uint16<R>
  : T extends Uint32  ? CommonType_Uint32<R>
  : T extends Uint64  ? CommonType_Uint64<R>
  : T extends Float32 ? CommonType_Float32<R>
  : T extends Float64 ? CommonType_Float64<R>
  : never;
// clang-format on

export type SeriesType<T extends DataType> = {
  [TypeId.EMPTY]: never,  // TODO
  [TypeId.INT8]: Int8Series,
  [TypeId.INT16]: Int16Series,
  [TypeId.INT32]: Int32Series,
  [TypeId.INT64]: Int64Series,
  [TypeId.UINT8]: Uint8Series,
  [TypeId.UINT16]: Uint16Series,
  [TypeId.UINT32]: Uint32Series,
  [TypeId.UINT64]: Uint64Series,
  [TypeId.FLOAT32]: Float32Series,
  [TypeId.FLOAT64]: Float64Series,
  [TypeId.BOOL8]: Bool8Series,
  [TypeId.TIMESTAMP_DAYS]: never,          // TODO
  [TypeId.TIMESTAMP_SECONDS]: never,       // TODO
  [TypeId.TIMESTAMP_MILLISECONDS]: never,  // TODO
  [TypeId.TIMESTAMP_MICROSECONDS]: never,  // TODO
  [TypeId.TIMESTAMP_NANOSECONDS]: never,   // TODO
  [TypeId.DURATION_DAYS]: never,           // TODO
  [TypeId.DURATION_SECONDS]: never,        // TODO
  [TypeId.DURATION_MILLISECONDS]: never,   // TODO
  [TypeId.DURATION_MICROSECONDS]: never,   // TODO
  [TypeId.DURATION_NANOSECONDS]: never,    // TODO
  [TypeId.DICTIONARY32]: never,            // TODO
  [TypeId.STRING]: StringSeries,
  [TypeId.LIST]: never,       // TODO
  [TypeId.DECIMAL32]: never,  // TODO
  [TypeId.DECIMAL64]: never,  // TODO
}[T['id']];
