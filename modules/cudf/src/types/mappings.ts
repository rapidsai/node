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

/** @ignore */
export type ArrowToCUDFType<T extends arrow.DataType> = {
  [arrow.Type.NONE]: never,
  [arrow.Type.Null]: never,
  [arrow.Type.Int]: never,
  [arrow.Type.Float]: never,
  [arrow.Type.Binary]: never,
  [arrow.Type.Bool]: Bool8,
  [arrow.Type.Utf8]: Utf8String,
  [arrow.Type.Decimal]: never,
  [arrow.Type.Date]: never,
  [arrow.Type.Time]: never,
  [arrow.Type.Timestamp]: never,
  [arrow.Type.Interval]: never,
  [arrow.Type.List]: List<T extends arrow.List ? ArrowToCUDFType<T['valueType']>: any>,
  [arrow.Type.Struct]:
    Struct<T extends arrow.Struct
                       ? {[P in keyof T['dataTypes']]: ArrowToCUDFType<T['dataTypes'][P]>}
                       : any>,
  [arrow.Type.Union]: never,
  [arrow.Type.FixedSizeBinary]: never,
  [arrow.Type.FixedSizeList]: never,
  [arrow.Type.Map]: never,
  [arrow.Type.Dictionary]: never,
  [arrow.Type.Int8]: Int8,
  [arrow.Type.Int16]: Int16,
  [arrow.Type.Int32]: Int32,
  [arrow.Type.Int64]: Int64,
  [arrow.Type.Uint8]: Uint8,
  [arrow.Type.Uint16]: Uint16,
  [arrow.Type.Uint32]: Uint32,
  [arrow.Type.Uint64]: Uint64,
  [arrow.Type.Float16]: never,
  [arrow.Type.Float32]: Float32,
  [arrow.Type.Float64]: Float64,
  [arrow.Type.DateDay]: never,
  [arrow.Type.DateMillisecond]: never,
  [arrow.Type.TimestampSecond]: never,
  [arrow.Type.TimestampMillisecond]: never,
  [arrow.Type.TimestampMicrosecond]: never,
  [arrow.Type.TimestampNanosecond]: never,
  [arrow.Type.TimeSecond]: never,
  [arrow.Type.TimeMillisecond]: never,
  [arrow.Type.TimeMicrosecond]: never,
  [arrow.Type.TimeNanosecond]: never,
  [arrow.Type.DenseUnion]: never,
  [arrow.Type.SparseUnion]: never,
  [arrow.Type.IntervalDayTime]: never,
  [arrow.Type.IntervalYearMonth]: never,
}[T['typeId']];

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
    // public visitNull                 <T extends arrow.Null>(_: T) { return new Null; }
    public visitBool                 <T extends arrow.Bool>(_: T) { return new Bool8; }
    public visitInt8                 <T extends arrow.Int8>(_: T) { return new Int8; }
    public visitInt16                <T extends arrow.Int16>(_: T) { return new Int16; }
    public visitInt32                <T extends arrow.Int32>(_: T) { return new Int32; }
    public visitInt64                <T extends arrow.Int64>(_: T) { return new Int64; }
    public visitUint8                <T extends arrow.Uint8>(_: T) { return new Uint8; }
    public visitUint16               <T extends arrow.Uint16>(_: T) { return new Uint16; }
    public visitUint32               <T extends arrow.Uint32>(_: T) { return new Uint32; }
    public visitUint64               <T extends arrow.Uint64>(_: T) { return new Uint64; }
    // public visitFloat16              <T extends arrow.Float16>(_: T) { return new Float16; }
    public visitFloat32              <T extends arrow.Float32>(_: T) { return new Float32; }
    public visitFloat64              <T extends arrow.Float64>(_: T) { return new Float64; }
    public visitUtf8                 <T extends arrow.Utf8>(_: T) { return new Utf8String; }
    // public visitBinary               <T extends arrow.Binary>(_: T) { return new Binary; }
    // public visitFixedSizeBinary      <T extends arrow.FixedSizeBinary>(_: T) { return new FixedSizeBinary(_); }
    // public visitDateDay              <T extends arrow.DateDay>(_: T) { return new DateDay; }
    // public visitDateMillisecond      <T extends arrow.DateMillisecond>(_: T) { return new DateMillisecond; }
    // public visitTimestampSecond      <T extends arrow.TimestampSecond>(_: T) { return new TimestampSecond; }
    // public visitTimestampMillisecond <T extends arrow.TimestampMillisecond>(_: T) { return new TimestampMillisecond; }
    // public visitTimestampMicrosecond <T extends arrow.TimestampMicrosecond>(_: T) { return new TimestampMicrosecond; }
    // public visitTimestampNanosecond  <T extends arrow.TimestampNanosecond>(_: T) { return new TimestampNanosecond; }
    // public visitTimeSecond           <T extends arrow.TimeSecond>(_: T) { return new TimeSecond; }
    // public visitTimeMillisecond      <T extends arrow.TimeMillisecond>(_: T) { return new TimeMillisecond; }
    // public visitTimeMicrosecond      <T extends arrow.TimeMicrosecond>(_: T) { return new TimeMicrosecond; }
    // public visitTimeNanosecond       <T extends arrow.TimeNanosecond>(_: T) { return new TimeNanosecond; }
    // public visitDecimal              <T extends arrow.Decimal>(_: T) { return new Decimal(_); }
    public visitList                 <T extends arrow.List>(_: T) { return new List(_.children[0]); }
    public visitStruct               <T extends arrow.Struct>(_: T) { return new Struct(_.children); }
    // public visitDenseUnion           <T extends arrow.DenseUnion>(_: T) { return new DenseUnion(_); }
    // public visitSparseUnion          <T extends arrow.SparseUnion>(_: T) { return new SparseUnion(_); }
    // public visitDictionary           <T extends arrow.Dictionary>(_: T) { return new Dictionary(_); }
    // public visitIntervalDayTime      <T extends arrow.IntervalDayTime>(_: T) { return new IntervalDayTime(_); }
    // public visitIntervalYearMonth    <T extends arrow.IntervalYearMonth>(_: T) { return new IntervalYearMonth(_); }
    // public visitFixedSizeList        <T extends arrow.FixedSizeList>(_: T) { return new FixedSizeList(_); }
    // public visitMap                  <T extends arrow.Map>(_: T) { return new Map(_); }
  }
  /* eslint-enable @typescript-eslint/no-unused-vars */
  // clang-format on
  const visitor = new ArrowToCUDFTypeVisitor();
  return function arrowToCUDFType<T extends arrow.DataType>(type: T): ArrowToCUDFType<T> {
    return visitor.visit(type);
  };
})();
