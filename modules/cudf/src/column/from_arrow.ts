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
import {
  Bool8,
  Categorical,
  Float32,
  Float64,
  Int16,
  Int32,
  Int64,
  Int8,
  List,
  Struct,
  TimestampDay,
  TimestampMicrosecond,
  TimestampMillisecond,
  TimestampNanosecond,
  TimestampSecond,
  Uint16,
  Uint32,
  Uint64,
  Uint8,
  Utf8String,
} from '../types/dtypes';
import {ArrowToCUDFType} from '../types/mappings';

/** @ignore */
interface DataToColumnVisitor extends arrow.Visitor {
  visit<T extends arrow.DataType>(node: arrow.Data<T>): Column<ArrowToCUDFType<T>>;
  visitMany<T extends arrow.DataType>(nodes: readonly arrow.Data<T>[]):
    Column<ArrowToCUDFType<T>>[];
  getVisitFn<T extends arrow.DataType>(node: arrow.Data<T>): () => Column<ArrowToCUDFType<T>>;
}

class DataToColumnVisitor extends arrow.Visitor {
  // visitNull<T extends arrow.Null>(data: arrow.Data<T>) {}
  visitBool<T extends arrow.Bool>(data: arrow.Data<T>) {
    const {values, nullBitmap: nullMask} = data;
    return new Column({
      type: new Bool8,
      data:
        // eslint-disable-next-line @typescript-eslint/unbound-method
        new Uint8Array(new arrow.util.BitIterator(values, 0, data.length, null, arrow.util.getBit)),
      nullMask
    });
  }
  visitInt8<T extends arrow.Int8>({length, values: data, nullBitmap: nullMask}: arrow.Data<T>) {
    return new Column({type: new Int8, length, data: data.subarray(0, length), nullMask});
  }
  visitInt16<T extends arrow.Int16>({length, values: data, nullBitmap: nullMask}: arrow.Data<T>) {
    return new Column({type: new Int16, length, data: data.subarray(0, length), nullMask});
  }
  visitInt32<T extends arrow.Int32>({length, values: data, nullBitmap: nullMask}: arrow.Data<T>) {
    return new Column({type: new Int32, length, data: data.subarray(0, length), nullMask});
  }
  visitInt64<T extends arrow.Int64>({length, values: data, nullBitmap: nullMask}: arrow.Data<T>) {
    return new Column({type: new Int64, length, data: data.subarray(0, length * 2), nullMask});
  }
  visitUint8<T extends arrow.Uint8>({length, values: data, nullBitmap: nullMask}: arrow.Data<T>) {
    return new Column({type: new Uint8, length, data: data.subarray(0, length), nullMask});
  }
  visitUint16<T extends arrow.Uint16>({length, values: data, nullBitmap: nullMask}: arrow.Data<T>) {
    return new Column({type: new Uint16, length, data: data.subarray(0, length), nullMask});
  }
  visitUint32<T extends arrow.Uint32>({length, values: data, nullBitmap: nullMask}: arrow.Data<T>) {
    return new Column({type: new Uint32, length, data: data.subarray(0, length), nullMask});
  }
  visitUint64<T extends arrow.Uint64>({length, values: data, nullBitmap: nullMask}: arrow.Data<T>) {
    return new Column({type: new Uint64, length, data: data.subarray(0, length * 2), nullMask});
  }
  // visitFloat16<T extends arrow.Float16>(data: arrow.Data<T>) {}
  visitFloat32<T extends arrow.Float32>({length, values: data, nullBitmap: nullMask}:
                                          arrow.Data<T>) {
    return new Column({type: new Float32, length, data: data.subarray(0, length), nullMask});
  }
  visitFloat64<T extends arrow.Float64>({length, values: data, nullBitmap: nullMask}:
                                          arrow.Data<T>) {
    return new Column({type: new Float64, length, data: data.subarray(0, length), nullMask});
  }
  visitUtf8<T extends arrow.Utf8>({length, values, valueOffsets, nullBitmap: nullMask}:
                                    arrow.Data<T>) {
    return new Column({
      length,
      type: new Utf8String,
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
  // visitBinary<T extends arrow.Binary>(data: arrow.Data<T>) {}
  // visitFixedSizeBinary<T extends arrow.FixedSizeBinary>(data: arrow.Data<T>) {}
  // visitDate<T extends arrow.Date_>(data: arrow.Data<T>) {}
  visitDateDay<T extends arrow.DateDay>({length, values: data, nullBitmap: nullMask}:
                                          arrow.Data<T>) {
    return new Column({type: new TimestampDay, length, data: data.subarray(0, length), nullMask});
  }
  visitDateMillisecond<T extends arrow.DateMillisecond>(
    {length, values: data, nullBitmap: nullMask}: arrow.Data<T>) {
    return new Column(
      {type: new TimestampMillisecond, length, data: data.subarray(0, length), nullMask});
  }
  visitTimestampSecond<T extends arrow.TimestampSecond>(
    {length, values: data, nullBitmap: nullMask}: arrow.Data<T>) {
    return new Column(
      {type: new TimestampSecond, length, data: data.subarray(0, length * 2), nullMask});
  }
  visitTimestampMillisecond<T extends arrow.TimestampMillisecond>(
    {length, values: data, nullBitmap: nullMask}: arrow.Data<T>) {
    return new Column(
      {type: new TimestampMillisecond, length, data: data.subarray(0, length * 2), nullMask});
  }
  visitTimestampMicrosecond<T extends arrow.TimestampMicrosecond>(
    {length, values: data, nullBitmap: nullMask}: arrow.Data<T>) {
    return new Column(
      {type: new TimestampMicrosecond, length, data: data.subarray(0, length * 2), nullMask});
  }
  visitTimestampNanosecond<T extends arrow.TimestampNanosecond>(
    {length, values: data, nullBitmap: nullMask}: arrow.Data<T>) {
    return new Column(
      {type: new TimestampNanosecond, length, data: data.subarray(0, length * 2), nullMask});
  }
  // visitTimeSecond<T extends arrow.TimeSecond>(data: arrow.Data<T>) {}
  // visitTimeMillisecond<T extends arrow.TimeMillisecond>(data: arrow.Data<T>) {}
  // visitTimeMicrosecond<T extends arrow.TimeMicrosecond>(data: arrow.Data<T>) {}
  // visitTimeNanosecond<T extends arrow.TimeNanosecond>(data: arrow.Data<T>) {}
  // visitDecimal<T extends arrow.Decimal>(data: arrow.Data<T>) {}
  visitList<T extends arrow.List>(data: arrow.Data<T>) {
    const {type, length, valueOffsets, nullBitmap: nullMask} = data;
    const offsets =
      new Column({type: new Int32, length: length + 1, data: valueOffsets.subarray(0, length + 1)});
    const elements = this.visit(data.children[0] as arrow.Data<T['valueType']>);
    return new Column({
      length,
      type: new List(type.children[0].clone({type: elements.type, nullable: elements.nullable})),
      nullMask,
      children: [
        offsets,
        elements,
      ]
    });
  }
  visitStruct<T extends arrow.Struct>(data: arrow.Data<T>) {
    const {type, length, nullBitmap: nullMask} = data;
    const children = type.children.map((_, i) => this.visit(data.children[i]));
    return new Column({
      length,
      type: new Struct(children.map(
        (child, i) => type.children[i].clone({type: child.type, nullable: child.nullable}))),
      nullMask,
      children,
    });
  }
  // visitDenseUnion<T extends arrow.DenseUnion>(data: arrow.Data<T>) {}
  // visitSparseUnion<T extends arrow.SparseUnion>(data: arrow.Data<T>) {}
  visitDictionary<T extends arrow.Dictionary>(data: arrow.Data<T>) {
    const {type, length, nullBitmap: nullMask} = data;
    const codes = this.visit(data.clone(type.indices)).cast(new Uint32);
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    const categories = fromArrow(data.dictionary!);
    return new Column(
      {length, type: new Categorical(categories.type), nullMask, children: [codes, categories]});
  }
  // visitIntervalDayTime<T extends arrow.IntervalDayTime>(data: arrow.Data<T>) {}
  // visitIntervalYearMonth<T extends arrow.IntervalYearMonth>(data: arrow.Data<T>) {}
  // visitFixedSizeList<T extends arrow.FixedSizeList>(data: arrow.Data<T>) {}
  // visitMap<T extends arrow.Map_>(data: arrow.Data<T>) {}
}

const visitor = new DataToColumnVisitor();

export function fromArrow<T extends arrow.DataType>(vector: arrow.Vector<T>):
  Column<ArrowToCUDFType<T>> {
  const cols = visitor.visitMany(vector.data);
  if (cols.length === 1) { return cols[0]; }
  return Table.concat(cols.map((col) => new Table({columns: [col]}))).getColumnByIndex(0);
  // return visitor.visit(vector);
}
