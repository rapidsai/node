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

import * as arrow from 'apache-arrow';

import {Column} from '../column';
import {Table} from '../table';
import {TypeMap} from './mappings';

export type FloatingPoint = Float32|Float64;
export type IndexType     = Int8|Int16|Int32|Uint8|Uint16|Uint32;
export type Integral      = IndexType|Int64|Uint64;
export type Numeric       = Integral|FloatingPoint|Bool8;
export type Timestamp =
  TimestampDay|TimestampSecond|TimestampMillisecond|TimestampMicrosecond|TimestampNanosecond;
export type DataType = Numeric|Utf8String|List|Struct|Timestamp|Categorical;

export interface Int8 extends arrow.Int8 {
  scalarType: number;
  readonly BYTES_PER_ELEMENT: number;
}
export class Int8 extends arrow.Int8 {}
(Int8.prototype as any).BYTES_PER_ELEMENT = 1;

export interface Int16 extends arrow.Int16 {
  scalarType: number;
  readonly BYTES_PER_ELEMENT: number;
}
export class Int16 extends arrow.Int16 {}
(Int16.prototype as any).BYTES_PER_ELEMENT = 2;

export interface Int32 extends arrow.Int32 {
  scalarType: number;
  readonly BYTES_PER_ELEMENT: number;
}
export class Int32 extends arrow.Int32 {}
(Int32.prototype as any).BYTES_PER_ELEMENT = 4;

export interface Int64 extends arrow.Int64 {
  TValue: bigint;
  scalarType: bigint;
  readonly BYTES_PER_ELEMENT: number;
}
export class Int64 extends arrow.Int64 {}
(Int64.prototype as any).BYTES_PER_ELEMENT = 8;

export interface Uint8 extends arrow.Uint8 {
  scalarType: number;
  readonly BYTES_PER_ELEMENT: number;
}
export class Uint8 extends arrow.Uint8 {}
(Uint8.prototype as any).BYTES_PER_ELEMENT = 1;

export interface Uint16 extends arrow.Uint16 {
  scalarType: number;
  readonly BYTES_PER_ELEMENT: number;
}
export class Uint16 extends arrow.Uint16 {}
(Uint16.prototype as any).BYTES_PER_ELEMENT = 2;

export interface Uint32 extends arrow.Uint32 {
  scalarType: number;
  readonly BYTES_PER_ELEMENT: number;
}
export class Uint32 extends arrow.Uint32 {}
(Uint32.prototype as any).BYTES_PER_ELEMENT = 4;

export interface Uint64 extends arrow.Uint64 {
  TValue: bigint;
  scalarType: bigint;
  readonly BYTES_PER_ELEMENT: number;
}
export class Uint64 extends arrow.Uint64 {}
(Uint64.prototype as any).BYTES_PER_ELEMENT = 8;

export interface Float32 extends arrow.Float32 {
  scalarType: number;
  readonly BYTES_PER_ELEMENT: number;
}
export class Float32 extends arrow.Float32 {}
(Float32.prototype as any).BYTES_PER_ELEMENT = 4;

export interface Float64 extends arrow.Float64 {
  scalarType: number;
  readonly BYTES_PER_ELEMENT: number;
}
export class Float64 extends arrow.Float64 {}
(Float64.prototype as any).BYTES_PER_ELEMENT = 8;

export interface Bool8 extends arrow.Bool {
  scalarType: boolean;
  readonly BYTES_PER_ELEMENT: number;
}
export class Bool8 extends arrow.Bool {}
(Bool8.prototype as any).BYTES_PER_ELEMENT = 1;

export interface Utf8String extends arrow.Utf8 {
  scalarType: string;
}
export class Utf8String extends arrow.Utf8 {}

export interface List<T extends DataType = any> extends arrow.List<T> {
  scalarType: Column<T>;
}
export class List<T extends DataType = any> extends arrow.List<T> {}

export interface Struct<T extends TypeMap = any> extends arrow.Struct<T> {
  scalarType: Table;
}
export class Struct<T extends TypeMap = any> extends arrow.Struct<T> {}

export interface TimestampDay extends arrow.DateDay {
  scalarType: Date;
}
export class TimestampDay extends arrow.DateDay {}

export interface TimestampSecond extends arrow.TimestampSecond {
  scalarType: Date;
}
export class TimestampSecond extends arrow.TimestampSecond {}

export interface TimestampMillisecond extends arrow.TimestampMillisecond {
  scalarType: Date;
}
export class TimestampMillisecond extends arrow.TimestampMillisecond {}

export interface TimestampMicrosecond extends arrow.TimestampMicrosecond {
  scalarType: Date;
}
export class TimestampMicrosecond extends arrow.TimestampMicrosecond {}

export interface TimestampNanosecond extends arrow.TimestampNanosecond {
  scalarType: Date;
}
export class TimestampNanosecond extends arrow.TimestampNanosecond {}

export interface Categorical<T extends DataType = any> extends arrow.Dictionary<T, Uint32> {
  scalarType: T['scalarType'];
}
export class Categorical<T extends DataType = any> extends arrow.Dictionary<T, Uint32> {
  constructor(dictionary: T, _id?: number|null, isOrdered?: boolean|null) {
    // we are overriding the id here so that Arrow dictionaries will always compare
    super(dictionary, new Uint32, 0, isOrdered);
  }
}

export const FloatTypes = [new Float32, new Float64];

export const IntegralTypes = [
  new Int8,
  new Int16,
  new Int32,
  new Int64,
  new Uint8,
  new Uint16,
  new Uint32,
  new Uint64,
];

export const NumericTypes = [new Bool8, ...FloatTypes, ...IntegralTypes];
