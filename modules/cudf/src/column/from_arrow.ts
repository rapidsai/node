// Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
  // Binary,
  Bool8,
  // Date_,
  // Decimal,
  // DenseUnion,
  // Dictionary,
  // FixedSizeBinary,
  // FixedSizeList,
  // Float16,
  Float32,
  Float64,
  Int16,
  Int32,
  Int64,
  Int8,
  // IntervalDayTime,
  // IntervalYearMonth,
  List,
  Struct,
  TimestampDay,
  TimestampMicrosecond,
  TimestampMillisecond,
  TimestampNanosecond,
  TimestampSecond,
  // Map_,
  // Null,
  // SparseUnion,
  // Struct,
  // TimeMicrosecond,
  // TimeMillisecond,
  // TimeNanosecond,
  // TimeSecond,
  Uint16,
  Uint32,
  Uint64,
  Uint8,
  Utf8String,
} from '../types/dtypes';
import {ArrowToCUDFType} from '../types/mappings';

/** @ignore */
interface VectorToColumnVisitor extends arrow.Visitor {
  visit<T extends arrow.DataType>(node: arrow.Vector<T>): Column<ArrowToCUDFType<T>>;
  visitMany<T extends arrow.DataType>(nodes: arrow.Vector<T>[]): Column<ArrowToCUDFType<T>>[];
  getVisitFn<T extends arrow.DataType>(node: arrow.Vector<T>): () => Column<ArrowToCUDFType<T>>;
}

class VectorToColumnVisitor extends arrow.Visitor {
  // visitNull<T extends arrow.Null>(vector: arrow.Vector<T>) {}
  visitBool<T extends arrow.Bool>(vector: arrow.Vector<T>) {
    const {nullBitmap: nullMask} = vector.data;
    const data                   = new Uint8Array([...vector].map((x) => x ? 1 : 0));
    return new Column({type: new Bool8, data, nullMask});
  }
  visitInt8<T extends arrow.Int8>({length,
                                   data: {values: data, nullBitmap: nullMask}}: arrow.Vector<T>) {
    return new Column({type: new Int8, length, data: data.subarray(0, length), nullMask});
  }
  visitInt16<T extends arrow.Int16>({length,
                                     data: {values: data, nullBitmap: nullMask}}: arrow.Vector<T>) {
    return new Column({type: new Int16, length, data: data.subarray(0, length), nullMask});
  }
  visitInt32<T extends arrow.Int32>({length,
                                     data: {values: data, nullBitmap: nullMask}}: arrow.Vector<T>) {
    return new Column({type: new Int32, length, data: data.subarray(0, length), nullMask});
  }
  visitInt64<T extends arrow.Int64>({length,
                                     data: {values: data, nullBitmap: nullMask}}: arrow.Vector<T>) {
    return new Column({type: new Int64, length, data: data.subarray(0, length * 2), nullMask});
  }
  visitUint8<T extends arrow.Uint8>({length,
                                     data: {values: data, nullBitmap: nullMask}}: arrow.Vector<T>) {
    return new Column({type: new Uint8, length, data: data.subarray(0, length), nullMask});
  }
  visitUint16<T extends arrow.Uint16>({length, data: {values: data, nullBitmap: nullMask}}:
                                        arrow.Vector<T>) {
    return new Column({type: new Uint16, length, data: data.subarray(0, length), nullMask});
  }
  visitUint32<T extends arrow.Uint32>({length, data: {values: data, nullBitmap: nullMask}}:
                                        arrow.Vector<T>) {
    return new Column({type: new Uint32, length, data: data.subarray(0, length), nullMask});
  }
  visitUint64<T extends arrow.Uint64>({length, data: {values: data, nullBitmap: nullMask}}:
                                        arrow.Vector<T>) {
    return new Column({type: new Uint64, length, data: data.subarray(0, length * 2), nullMask});
  }
  // visitFloat16<T extends arrow.Float16>(vector: arrow.Vector<T>) {}
  visitFloat32<T extends arrow.Float32>({length, data: {values: data, nullBitmap: nullMask}}:
                                          arrow.Vector<T>) {
    return new Column({type: new Float32, length, data: data.subarray(0, length), nullMask});
  }
  visitFloat64<T extends arrow.Float64>({length, data: {values: data, nullBitmap: nullMask}}:
                                          arrow.Vector<T>) {
    return new Column({type: new Float64, length, data: data.subarray(0, length), nullMask});
  }
  visitUtf8<T extends arrow.Utf8>({length,
                                   data: {values, valueOffsets, nullBitmap}}: arrow.Vector<T>) {
    return new Column({
      length,
      type: new Utf8String,
      nullMask: nullBitmap,
      children: [
        // offsets
        new Column(
          {type: new Int32, length: length + 1, data: valueOffsets.subarray(0, length + 1)}),
        // data
        new Column({
          type: new Uint8,
          length: valueOffsets[length],
          data: values.subarray(0, valueOffsets[length])
        }),
      ]
    });
  }
  // visitBinary<T extends arrow.Binary>(vector: arrow.Vector<T>) {}
  // visitFixedSizeBinary<T extends arrow.FixedSizeBinary>(vector: arrow.Vector<T>) {}
  // visitDate<T extends arrow.Date_>(vector: arrow.Vector<T>) {}
  visitDateDay<T extends arrow.DateDay>({length, data: {values: data, nullBitmap: nullMask}}:
                                          arrow.Vector<T>) {
    return new Column({type: new TimestampDay, length, data: data.subarray(0, length), nullMask});
  }
  visitDateMillisecond<T extends arrow.DateMillisecond>(
    {length, data: {values: data, nullBitmap: nullMask}}: arrow.Vector<T>) {
    return new Column(
      {type: new TimestampMillisecond, length, data: data.subarray(0, length), nullMask});
  }
  visitTimestampSecond<T extends arrow.TimestampSecond>(
    {length, data: {values: data, nullBitmap: nullMask}}: arrow.Vector<T>) {
    return new Column(
      {type: new TimestampSecond, length, data: data.subarray(0, length), nullMask});
  }
  visitTimestampMillisecond<T extends arrow.TimestampMillisecond>(
    {length, data: {values: data, nullBitmap: nullMask}}: arrow.Vector<T>) {
    return new Column(
      {type: new TimestampMillisecond, length, data: data.subarray(0, length), nullMask});
  }
  visitTimestampMicrosecond<T extends arrow.TimestampMicrosecond>(
    {length, data: {values: data, nullBitmap: nullMask}}: arrow.Vector<T>) {
    return new Column(
      {type: new TimestampMicrosecond, length, data: data.subarray(0, length), nullMask});
  }
  visitTimestampNanosecond<T extends arrow.TimestampNanosecond>(
    {length, data: {values: data, nullBitmap: nullMask}}: arrow.Vector<T>) {
    return new Column(
      {type: new TimestampNanosecond, length, data: data.subarray(0, length), nullMask});
  }
  // visitTimeSecond<T extends arrow.TimeSecond>(vector: arrow.Vector<T>) {}
  // visitTimeMillisecond<T extends arrow.TimeMillisecond>(vector: arrow.Vector<T>) {}
  // visitTimeMicrosecond<T extends arrow.TimeMicrosecond>(vector: arrow.Vector<T>) {}
  // visitTimeNanosecond<T extends arrow.TimeNanosecond>(vector: arrow.Vector<T>) {}
  // visitDecimal<T extends arrow.Decimal>(vector: arrow.Vector<T>) {}
  visitList<T extends arrow.List>(vector: arrow.Vector<T>) {
    const {length, data: {valueOffsets, nullBitmap}} = vector;
    return new Column({
      length,
      type: new List(vector.type.children[0]),
      nullMask: nullBitmap,
      children: [
        // offsets
        new Column(
          {type: new Int32, length: length + 1, data: valueOffsets.subarray(0, length + 1)}),
        // elements
        this.visit(vector.getChildAt(0) as arrow.Vector<T['valueType']>),
      ]
    });
  }
  visitStruct<T extends arrow.Struct>(vector: arrow.Vector<T>) {
    const {length, data: {nullBitmap}} = vector;
    return new Column({
      length,
      type: new Struct(vector.type.children),
      nullMask: nullBitmap,
      children: vector.type.children.map((_, i) => this.visit(vector.getChildAt(i) as arrow.Vector))
    });
  }
  // visitDenseUnion<T extends arrow.DenseUnion>(vector: arrow.Vector<T>) {}
  // visitSparseUnion<T extends arrow.SparseUnion>(vector: arrow.Vector<T>) {}
  // visitDictionary<T extends arrow.Dictionary>(vector: arrow.Vector<T>) {}
  // visitIntervalDayTime<T extends arrow.IntervalDayTime>(vector: arrow.Vector<T>) {}
  // visitIntervalYearMonth<T extends arrow.IntervalYearMonth>(vector: arrow.Vector<T>) {}
  // visitFixedSizeList<T extends arrow.FixedSizeList>(vector: arrow.Vector<T>) {}
  // visitMap<T extends arrow.Map_>(vector: arrow.Vector<T>) {}
}

const visitor = new VectorToColumnVisitor();

export function fromArrow<T extends arrow.DataType>(vector: arrow.Vector<T>):
  Column<ArrowToCUDFType<T>> {
  return visitor.visit(vector);
}
