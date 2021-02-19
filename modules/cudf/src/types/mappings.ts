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

import * as arrow from 'apache-arrow';

import {Column} from '../column';

import {
  Bool8,
  DataType,
  Float32,
  Float64,
  Int16,
  Int32,
  Int64,
  Int8,
  List,
  Numeric,
  Struct,
  Uint16,
  Uint32,
  Uint64,
  Uint8,
  Utf8String,
} from './dtypes';

export enum Interpolation
{
  linear,    ///< Linear interpolation between i and j
  lower,     ///< Lower data point (i)
  higher,    ///< Higher data point (j)
  midpoint,  ///< (i + j)/2
  nearest    ///< i or j, whichever is nearest
}

export type TypeMap = {
  [key: string]: DataType
};

export type ColumnsMap<T extends TypeMap> = {
  [P in keyof T]: Column<T[P]>
};

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
export type CommonType<T extends DataType, R extends Numeric> =
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

// clang-format off
/** @ignore */
export type ArrowToCUDFType<T extends arrow.DataType> =
//  T extends arrow.Null ? never :   // TODO
 T extends arrow.Int8 ? Int8 : 
 T extends arrow.Int16 ? Int16 : 
 T extends arrow.Int32 ? Int32 : 
 T extends arrow.Int64 ? Int64 : 
 T extends arrow.Uint8 ? Uint8 : 
 T extends arrow.Uint16 ? Uint16 : 
 T extends arrow.Uint32 ? Uint32 : 
 T extends arrow.Uint64 ? Uint64 : 
//  T extends arrow.Int ? never : 
//  T extends arrow.Float16 ? never : 
 T extends arrow.Float32 ? Float32 : 
 T extends arrow.Float64 ? Float64 : 
//  T extends arrow.Float ? never : 
//  T extends arrow.Binary ? never : 
 T extends arrow.Utf8 ? Utf8String : 
 T extends arrow.Bool ? Bool8 : 
//  T extends arrow.Decimal ? never :                // TODO
//  T extends arrow.DateDay ? never :                // TODO
//  T extends arrow.DateMillisecond ? never :        // TODO
//  T extends arrow.Date_ ? never :                  // TODO
//  T extends arrow.TimeSecond ? never :             // TODO
//  T extends arrow.TimeMillisecond ? never :        // TODO
//  T extends arrow.TimeMicrosecond ? never :        // TODO
//  T extends arrow.TimeNanosecond ? never :         // TODO
//  T extends arrow.Time ? never :                   // TODO
//  T extends arrow.TimestampSecond ? never :        // TODO
//  T extends arrow.TimestampMillisecond ? never :   // TODO
//  T extends arrow.TimestampMicrosecond ? never :   // TODO
//  T extends arrow.TimestampNanosecond ? never :    // TODO
//  T extends arrow.Timestamp ? never :              // TODO
//  T extends arrow.IntervalDayTime ? never :        // TODO
//  T extends arrow.IntervalYearMonth ? never :      // TODO
//  T extends arrow.Interval ? never :               // TODO
 T extends arrow.List ? T extends List ? T : List<ArrowToCUDFType<T['valueType']>> :
 T extends arrow.Struct ? T extends Struct ? T : Struct<{[P in keyof T['dataTypes']]: ArrowToCUDFType<T['dataTypes'][P]>}> :
//  T extends arrow.Union ? never :
//  T extends arrow.DenseUnion ? never :
//  T extends arrow.SparseUnion ? never :
//  T extends arrow.FixedSizeBinary ? never :
//  T extends arrow.FixedSizeList ? never :
//  T extends arrow.Map_ ? never :
//  T extends arrow.Dictionary ? never :
 never;
// clang-format on

export const arrowToCUDFType = (() => {
  interface ArrowToCUDFTypeVisitor extends arrow.Visitor {
    visit<T extends arrow.DataType>(node: T): ArrowToCUDFType<T>;
    visitMany<T extends arrow.DataType>(nodes: T[]): ArrowToCUDFType<T>[];
    getVisitFn<T extends arrow.DataType>(node: T): () => ArrowToCUDFType<T>;
  }
  // clang-format off
  /* eslint-disable @typescript-eslint/no-unused-vars */
  class ArrowToCUDFTypeVisitor extends arrow.Visitor {
    getVisitFn<T extends arrow.DataType>(type: T): (type: T) => ArrowToCUDFType<T> {
      if (!(type instanceof arrow.DataType)) {
        return super.getVisitFn({
          ...(type as any),
          __proto__: arrow.DataType.prototype
        });
      }
      return super.getVisitFn(type);
    }
    // public visitNull                 <T extends arrow.Null>(type: T) { return new Null; }
    public visitBool                 <T extends arrow.Bool>(_type: T) { return new Bool8; }
    public visitInt8                 <T extends arrow.Int8>(_type: T) { return new Int8; }
    public visitInt16                <T extends arrow.Int16>(_type: T) { return new Int16; }
    public visitInt32                <T extends arrow.Int32>(_type: T) { return new Int32; }
    public visitInt64                <T extends arrow.Int64>(_type: T) { return new Int64; }
    public visitUint8                <T extends arrow.Uint8>(_type: T) { return new Uint8; }
    public visitUint16               <T extends arrow.Uint16>(_type: T) { return new Uint16; }
    public visitUint32               <T extends arrow.Uint32>(_type: T) { return new Uint32; }
    public visitUint64               <T extends arrow.Uint64>(_type: T) { return new Uint64; }
    // public visitFloat16              <T extends arrow.Float16>(_type: T) { return new Float16; }
    public visitFloat32              <T extends arrow.Float32>(_type: T) { return new Float32; }
    public visitFloat64              <T extends arrow.Float64>(_type: T) { return new Float64; }
    public visitUtf8                 <T extends arrow.Utf8>(_type: T) { return new Utf8String; }
    // public visitBinary               <T extends arrow.Binary>(_type: T) { return new Binary; }
    // public visitFixedSizeBinary      <T extends arrow.FixedSizeBinary>(type: T) { return new FixedSizeBinary(type); }
    // public visitDateDay              <T extends arrow.DateDay>(_type: T) { return new DateDay; }
    // public visitDateMillisecond      <T extends arrow.DateMillisecond>(_type: T) { return new DateMillisecond; }
    // public visitTimestampSecond      <T extends arrow.TimestampSecond>(_type: T) { return new TimestampSecond; }
    // public visitTimestampMillisecond <T extends arrow.TimestampMillisecond>(_type: T) { return new TimestampMillisecond; }
    // public visitTimestampMicrosecond <T extends arrow.TimestampMicrosecond>(_type: T) { return new TimestampMicrosecond; }
    // public visitTimestampNanosecond  <T extends arrow.TimestampNanosecond>(_type: T) { return new TimestampNanosecond; }
    // public visitTimeSecond           <T extends arrow.TimeSecond>(_type: T) { return new TimeSecond; }
    // public visitTimeMillisecond      <T extends arrow.TimeMillisecond>(_type: T) { return new TimeMillisecond; }
    // public visitTimeMicrosecond      <T extends arrow.TimeMicrosecond>(_type: T) { return new TimeMicrosecond; }
    // public visitTimeNanosecond       <T extends arrow.TimeNanosecond>(_type: T) { return new TimeNanosecond; }
    // public visitDecimal              <T extends arrow.Decimal>(_type: T) { return new Decimal(type); }
    public visitList                 <T extends arrow.List>(type: T) {
      const { name, type: childType } = type.children[0];
      return new List(arrow.Field.new({ name, type: this.visit(childType) }));
    }
    public visitStruct               <T extends arrow.Struct>(type: T) {
      return new Struct(type.children.map(({ name, type: childType }) => {
        return arrow.Field.new({ name, type: this.visit(childType) });
      }));
    }
    // public visitDenseUnion           <T extends arrow.DenseUnion>(type: T) { return new DenseUnion(type); }
    // public visitSparseUnion          <T extends arrow.SparseUnion>(type: T) { return new SparseUnion(type); }
    // public visitDictionary           <T extends arrow.Dictionary>(type: T) { return new Dictionary(type); }
    // public visitIntervalDayTime      <T extends arrow.IntervalDayTime>(type: T) { return new IntervalDayTime; }
    // public visitIntervalYearMonth    <T extends arrow.IntervalYearMonth>(type: T) { return new IntervalYearMonth; }
    // public visitFixedSizeList        <T extends arrow.FixedSizeList>(type: T) { return new FixedSizeList(type); }
    // public visitMap                  <T extends arrow.Map>(type: T) { return new Map(type); }
  }
  /* eslint-enable @typescript-eslint/no-unused-vars */
  // clang-format on
  const visitor = new ArrowToCUDFTypeVisitor();
  return function arrowToCUDFType<T extends arrow.DataType>(type: T): ArrowToCUDFType<T> {
    return visitor.visit(type);
  };
})();
