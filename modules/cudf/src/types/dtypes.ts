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

import {
  Float32Buffer,
  Float64Buffer,
  Int16Buffer,
  Int32Buffer,
  Int64Buffer,
  Int8Buffer,
  Uint16Buffer,
  Uint32Buffer,
  Uint64Buffer,
  Uint8Buffer,
  Uint8ClampedBuffer,
} from '@rapidsai/cuda';
import * as arrow from 'apache-arrow';

import {Column} from '../column';
import {DataFrame, SeriesMap} from '../data_frame';
import {Scalar} from '../scalar';
import {Series} from '../series';
import {Table} from '../table';

import {ColumnsMap, TypeMap} from './mappings';

export type FloatingPoint = Float32|Float64;
export type IndexType     = Int8|Int16|Int32|Uint8|Uint16|Uint32;
export type Integral      = IndexType|Int64|Uint64;
export type Numeric       = Integral|FloatingPoint|Bool8;
export type Timestamp =
  TimestampDay|TimestampSecond|TimestampMillisecond|TimestampMicrosecond|TimestampNanosecond;
export type DataType = Numeric|Utf8String|List|Struct|Timestamp|Categorical;

const ab  = new ArrayBuffer(16);
const i8  = new Int8Array(ab);
const i16 = new Int16Array(ab);
const i32 = new Int32Array(ab);
const i64 = new BigInt64Array(ab);
const u8  = new Uint8Array(ab);
const u16 = new Uint16Array(ab);
const u32 = new Uint32Array(ab);
const u64 = new BigUint64Array(ab);
const f32 = new Float32Array(ab);
const f64 = new Float64Array(ab);

export interface Int8 extends arrow.Int8 {
  scalarType: number;
  readonly BYTES_PER_ELEMENT: number;
}
export class Int8 extends arrow.Int8 {
  scalar(value: any) {  //
    return new Scalar({type: new Int8, value: (i8[0] = value)});
  }
}
(Int8.prototype as any).BYTES_PER_ELEMENT = 1;

export interface Int16 extends arrow.Int16 {
  scalarType: number;
  readonly BYTES_PER_ELEMENT: number;
}
export class Int16 extends arrow.Int16 {
  scalar(value: any) {  //
    return new Scalar({type: new Int16, value: (i16[0] = value)});
  }
}
(Int16.prototype as any).BYTES_PER_ELEMENT = 2;

export interface Int32 extends arrow.Int32 {
  scalarType: number;
  readonly BYTES_PER_ELEMENT: number;
}
export class Int32 extends arrow.Int32 {
  scalar(value: any) {  //
    return new Scalar({type: new Int32, value: (i32[0] = value)});
  }
}
(Int32.prototype as any).BYTES_PER_ELEMENT = 4;

export interface Int64 extends arrow.Int64 {
  TValue: bigint;
  scalarType: bigint;
  readonly BYTES_PER_ELEMENT: number;
}
export class Int64 extends arrow.Int64 {
  scalar(value: any) {  //
    return new Scalar({type: new Int64, value: (i64[0] = value)});
  }
}
(Int64.prototype as any).BYTES_PER_ELEMENT = 8;

export interface Uint8 extends arrow.Uint8 {
  scalarType: number;
  readonly BYTES_PER_ELEMENT: number;
}
export class Uint8 extends arrow.Uint8 {
  scalar(value: any) {  //
    return new Scalar({type: new Uint8, value: (u8[0] = value)});
  }
}
(Uint8.prototype as any).BYTES_PER_ELEMENT = 1;

export interface Uint16 extends arrow.Uint16 {
  scalarType: number;
  readonly BYTES_PER_ELEMENT: number;
}
export class Uint16 extends arrow.Uint16 {
  scalar(value: any) {  //
    return new Scalar({type: new Uint16, value: (u16[0] = value)});
  }
}
(Uint16.prototype as any).BYTES_PER_ELEMENT = 2;

export interface Uint32 extends arrow.Uint32 {
  scalarType: number;
  readonly BYTES_PER_ELEMENT: number;
}
export class Uint32 extends arrow.Uint32 {
  scalar(value: any) {  //
    return new Scalar({type: new Uint32, value: (u32[0] = value)});
  }
}
(Uint32.prototype as any).BYTES_PER_ELEMENT = 4;

export interface Uint64 extends arrow.Uint64 {
  TValue: bigint;
  scalarType: bigint;
  readonly BYTES_PER_ELEMENT: number;
}
export class Uint64 extends arrow.Uint64 {
  scalar(value: any) {  //
    return new Scalar({type: new Uint64, value: (u64[0] = value)});
  }
}
(Uint64.prototype as any).BYTES_PER_ELEMENT = 8;

export interface Float32 extends arrow.Float32 {
  scalarType: number;
  readonly BYTES_PER_ELEMENT: number;
}
export class Float32 extends arrow.Float32 {
  scalar(value: any) {  //
    return new Scalar({type: new Float32, value: (f32[0] = value)});
  }
}
(Float32.prototype as any).BYTES_PER_ELEMENT = 4;

export interface Float64 extends arrow.Float64 {
  scalarType: number;
  readonly BYTES_PER_ELEMENT: number;
}
export class Float64 extends arrow.Float64 {
  scalar(value: any) {  //
    return new Scalar({type: new Float64, value: (f64[0] = value)});
  }
}
(Float64.prototype as any).BYTES_PER_ELEMENT = 8;

export interface Bool8 extends arrow.Bool {
  scalarType: boolean;
  readonly BYTES_PER_ELEMENT: number;
}
export class Bool8 extends arrow.Bool {
  scalar(value: any) {  //
    return new Scalar({type: new Bool8, value: (u8[0] = +!!value) === 1});
  }
}
(Bool8.prototype as any).BYTES_PER_ELEMENT = 1;

export interface Utf8String extends arrow.Utf8 {
  scalarType: string;
}
export class Utf8String extends arrow.Utf8 {
  scalar(value: any) {  //
    return new Scalar({type: new Utf8String, value: '' + <string>value});
  }
}

export interface List<T extends DataType = any> extends arrow.List<T> {
  childType: T;
  scalarType: Column<T>;
}
export class List<T extends DataType = any> extends arrow.List<T> {
  scalar(value: Int8Array|Int8Buffer): Scalar<List<Int8>>;
  scalar(value: Int16Array|Int16Buffer): Scalar<List<Int16>>;
  scalar(value: Int32Array|Int32Buffer): Scalar<List<Int32>>;
  scalar(value: Uint8Array|Uint8Buffer|Uint8ClampedArray|Uint8ClampedBuffer): Scalar<List<Uint8>>;
  scalar(value: Uint16Array|Uint16Buffer): Scalar<List<Uint16>>;
  scalar(value: Uint32Array|Uint32Buffer): Scalar<List<Uint32>>;
  scalar(value: BigUint64Array|Uint64Buffer): Scalar<List<Uint64>>;
  scalar(value: Float32Array|Float32Buffer): Scalar<List<Float32>>;
  scalar(value: (string|null|undefined)[]): Scalar<List<Utf8String>>;
  scalar(value: (number|null|undefined)[]|Float64Array|Float64Buffer): Scalar<List<Float64>>;
  scalar(value: (bigint|null|undefined)[]|BigInt64Array|Int64Buffer): Scalar<List<Int64>>;
  scalar(value: (boolean|null|undefined)[]): Scalar<List<Bool8>>;
  scalar(value: (Date|null|undefined)[]): Scalar<List<TimestampMillisecond>>;
  scalar(value: (string|null|undefined)[][]): Scalar<List<List<Utf8String>>>;
  scalar(value: (number|null|undefined)[][]): Scalar<List<List<Float64>>>;
  scalar(value: (bigint|null|undefined)[][]): Scalar<List<List<Int64>>>;
  scalar(value: (boolean|null|undefined)[][]): Scalar<List<List<Bool8>>>;
  scalar(value: (Date|null|undefined)[][]): Scalar<List<List<TimestampMillisecond>>>;
  scalar(value: any) {
    const {type, _col: col} = Series.new(value);
    return new Scalar({type: new List(new arrow.Field('0', type)), value: col});
  }
}

export interface Struct<T extends TypeMap = any> extends arrow.Struct<T> {
  childTypes: T;
  scalarType: {[P in keyof T]: T[P]['scalarType']};
}
export class Struct<T extends TypeMap = any> extends arrow.Struct<T> {
  scalar<T extends TypeMap>(value: DataFrame<T>): Scalar<Struct<T>>;
  scalar<T extends TypeMap>(value: SeriesMap<T>): Scalar<Struct<T>>;
  scalar<T extends TypeMap>(value: ColumnsMap<T>): Scalar<Struct<T>>;
  scalar(value: any) {
    if (value instanceof Table) {
      value = new DataFrame(Array.from({length: value.numColumns}, (_, i) => i)
                              .reduce((xs, x) => ({...xs, [x]: value.getColumnByIndex(x)}), {}));
    }
    const frame = value instanceof DataFrame ? value : new DataFrame(value);
    const type  = new Struct(frame.names.map((name) => arrow.Field.new(name, frame.types[name])));
    return new Scalar({type, value: frame.asTable()});
  }
}

export interface TimestampDay extends arrow.DateDay {
  scalarType: Date;
}
export class TimestampDay extends arrow.DateDay {
  scalar(value: any) {  //
    return new Scalar({type: new TimestampDay, value: new Date(value)});
  }
}

export interface TimestampSecond extends arrow.TimestampSecond {
  scalarType: Date;
}
export class TimestampSecond extends arrow.TimestampSecond {
  scalar(value: any) {  //
    return new Scalar({type: new TimestampSecond, value: new Date(value)});
  }
}

export interface TimestampMillisecond extends arrow.TimestampMillisecond {
  scalarType: Date;
}
export class TimestampMillisecond extends arrow.TimestampMillisecond {
  scalar(value: any) {  //
    return new Scalar({type: new TimestampMillisecond, value: new Date(value)});
  }
}

export interface TimestampMicrosecond extends arrow.TimestampMicrosecond {
  scalarType: Date;
}
export class TimestampMicrosecond extends arrow.TimestampMicrosecond {
  scalar(value: any) {  //
    return new Scalar({type: new TimestampMicrosecond, value: new Date(value)});
  }
}

export interface TimestampNanosecond extends arrow.TimestampNanosecond {
  scalarType: Date;
}
export class TimestampNanosecond extends arrow.TimestampNanosecond {
  scalar(value: any) {  //
    return new Scalar({type: new TimestampNanosecond, value: new Date(value)});
  }
}

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
