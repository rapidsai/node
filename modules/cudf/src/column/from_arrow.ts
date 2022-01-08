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
import {Int32, Uint32, Uint8} from '../types/dtypes';
import {arrowToCUDFType, ArrowToCUDFType} from '../types/mappings';

/** @ignore */
interface VectorToColumnVisitor extends arrow.Visitor {
  visit<T extends arrow.DataType>(node: arrow.Vector<T>): Column<ArrowToCUDFType<T>>;
  visitMany<T extends arrow.DataType>(nodes: arrow.Vector<T>[]): Column<ArrowToCUDFType<T>>[];
  getVisitFn<T extends arrow.DataType>(node: arrow.Vector<T>): () => Column<ArrowToCUDFType<T>>;
}

class VectorToColumnVisitor extends arrow.Visitor {
  // visitNull<T extends arrow.Null>(vector: arrow.Vector<T>) {}
  visitBool<T extends arrow.Bool>(vector: arrow.Vector<T>) {
    const {type, nullBitmap: nullMask} = vector.data;
    return new Column({type: arrowToCUDFType(type), data: new Uint8Array(vector), nullMask});
  }
  visitInt8<T extends arrow.Int8>({type, length, data: {values: data, nullBitmap: nullMask}}:
                                    arrow.Vector<T>) {
    return new Column(
      {type: arrowToCUDFType(type), length, data: data.subarray(0, length), nullMask});
  }
  visitInt16<T extends arrow.Int16>({type, length, data: {values: data, nullBitmap: nullMask}}:
                                      arrow.Vector<T>) {
    return new Column(
      {type: arrowToCUDFType(type), length, data: data.subarray(0, length), nullMask});
  }
  visitInt32<T extends arrow.Int32>({type, length, data: {values: data, nullBitmap: nullMask}}:
                                      arrow.Vector<T>) {
    return new Column(
      {type: arrowToCUDFType(type), length, data: data.subarray(0, length), nullMask});
  }
  visitInt64<T extends arrow.Int64>({type, length, data: {values: data, nullBitmap: nullMask}}:
                                      arrow.Vector<T>) {
    return new Column(
      {type: arrowToCUDFType(type), length, data: data.subarray(0, length * 2), nullMask});
  }
  visitUint8<T extends arrow.Uint8>({type, length, data: {values: data, nullBitmap: nullMask}}:
                                      arrow.Vector<T>) {
    return new Column(
      {type: arrowToCUDFType(type), length, data: data.subarray(0, length), nullMask});
  }
  visitUint16<T extends arrow.Uint16>({type, length, data: {values: data, nullBitmap: nullMask}}:
                                        arrow.Vector<T>) {
    return new Column(
      {type: arrowToCUDFType(type), length, data: data.subarray(0, length), nullMask});
  }
  visitUint32<T extends arrow.Uint32>({type, length, data: {values: data, nullBitmap: nullMask}}:
                                        arrow.Vector<T>) {
    return new Column(
      {type: arrowToCUDFType(type), length, data: data.subarray(0, length), nullMask});
  }
  visitUint64<T extends arrow.Uint64>({type, length, data: {values: data, nullBitmap: nullMask}}:
                                        arrow.Vector<T>) {
    return new Column(
      {type: arrowToCUDFType(type), length, data: data.subarray(0, length * 2), nullMask});
  }
  // visitFloat16<T extends arrow.Float16>(vector: arrow.Vector<T>) {}
  visitFloat32<T extends arrow.Float32>({type, length, data: {values: data, nullBitmap: nullMask}}:
                                          arrow.Vector<T>) {
    return new Column(
      {type: arrowToCUDFType(type), length, data: data.subarray(0, length), nullMask});
  }
  visitFloat64<T extends arrow.Float64>({type, length, data: {values: data, nullBitmap: nullMask}}:
                                          arrow.Vector<T>) {
    return new Column(
      {type: arrowToCUDFType(type), length, data: data.subarray(0, length), nullMask});
  }
  visitUtf8<T extends arrow.Utf8>(
    {type, length, data: {values, valueOffsets, nullBitmap: nullMask}}: arrow.Vector<T>) {
    return new Column({
      length,
      type: arrowToCUDFType(type),
      nullMask,
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
  visitDateDay<T extends arrow.DateDay>({type, length, data: {values: data, nullBitmap: nullMask}}:
                                          arrow.Vector<T>) {
    return new Column(
      {type: arrowToCUDFType(type), length, data: data.subarray(0, length), nullMask});
  }
  visitDateMillisecond<T extends arrow.DateMillisecond>(
    {type, length, data: {values: data, nullBitmap: nullMask}}: arrow.Vector<T>) {
    return new Column(
      {type: arrowToCUDFType(type), length, data: data.subarray(0, length), nullMask});
  }
  visitTimestampSecond<T extends arrow.TimestampSecond>(
    {type, length, data: {values: data, nullBitmap: nullMask}}: arrow.Vector<T>) {
    return new Column(
      {type: arrowToCUDFType(type), length, data: data.subarray(0, length * 2), nullMask});
  }
  visitTimestampMillisecond<T extends arrow.TimestampMillisecond>(
    {type, length, data: {values: data, nullBitmap: nullMask}}: arrow.Vector<T>) {
    return new Column(
      {type: arrowToCUDFType(type), length, data: data.subarray(0, length * 2), nullMask});
  }
  visitTimestampMicrosecond<T extends arrow.TimestampMicrosecond>(
    {type, length, data: {values: data, nullBitmap: nullMask}}: arrow.Vector<T>) {
    return new Column(
      {type: arrowToCUDFType(type), length, data: data.subarray(0, length * 2), nullMask});
  }
  visitTimestampNanosecond<T extends arrow.TimestampNanosecond>(
    {type, length, data: {values: data, nullBitmap: nullMask}}: arrow.Vector<T>) {
    return new Column(
      {type: arrowToCUDFType(type), length, data: data.subarray(0, length * 2), nullMask});
  }
  // visitTimeSecond<T extends arrow.TimeSecond>(vector: arrow.Vector<T>) {}
  // visitTimeMillisecond<T extends arrow.TimeMillisecond>(vector: arrow.Vector<T>) {}
  // visitTimeMicrosecond<T extends arrow.TimeMicrosecond>(vector: arrow.Vector<T>) {}
  // visitTimeNanosecond<T extends arrow.TimeNanosecond>(vector: arrow.Vector<T>) {}
  // visitDecimal<T extends arrow.Decimal>(vector: arrow.Vector<T>) {}
  visitList<T extends arrow.List>(vector: arrow.Vector<T>) {
    const {type, length, data: {valueOffsets, nullBitmap: nullMask}} = vector;
    return new Column({
      length,
      type: arrowToCUDFType(type),
      nullMask,
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
    const {type, length, data: {nullBitmap: nullMask}} = vector;
    return new Column({
      length,
      type: arrowToCUDFType(type),
      nullMask,
      children: type.children.map((_, i) => this.visit(vector.getChildAt(i) as arrow.Vector))
    });
  }
  // visitDenseUnion<T extends arrow.DenseUnion>(vector: arrow.Vector<T>) {}
  // visitSparseUnion<T extends arrow.SparseUnion>(vector: arrow.Vector<T>) {}
  visitDictionary<T extends arrow.Dictionary>(vector: arrow.Vector<T>) {
    const {type, length, data: {nullBitmap: nullMask}} = vector;
    const codes = this.visit(arrow.Vector.new(vector.data.clone(type.indices))).cast(new Uint32);
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    const categories = this.visit(vector.data.dictionary!);
    return new Column(
      {length, type: arrowToCUDFType(type), nullMask, children: [codes, categories]});
  }
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
