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

import {
  // Binary,
  Bool,
  DataType,
  // Date_,
  // DateDay,
  // DateMillisecond,
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
  // Map_,
  // Null,
  // SparseUnion,
  // Struct,
  // TimeMicrosecond,
  // TimeMillisecond,
  // TimeNanosecond,
  // TimeSecond,
  // TimestampMicrosecond,
  // TimestampMillisecond,
  // TimestampNanosecond,
  // TimestampSecond,
  Uint16,
  Uint32,
  Uint64,
  Uint8,
  Utf8,
  Vector,
  Visitor,
} from 'apache-arrow';

import {Column} from '../column';
import {ArrowToCUDFType, TypeId} from '../types';

/** @ignore */
interface VectorToColumnVisitor extends Visitor {
  visit<T extends DataType>(node: Vector<T>): Column<ArrowToCUDFType<T>>;
  visitMany<T extends DataType>(nodes: Vector<T>[]): Column<ArrowToCUDFType<T>>[];
  getVisitFn<T extends DataType>(node: Vector<T>): () => Column<ArrowToCUDFType<T>>;
}

class VectorToColumnVisitor extends Visitor {
  // visitNull<T extends Null>(vector: Vector<T>) {}
  visitBool<T extends Bool>(vector: Vector<T>) {
    const {nullBitmap: nullMask} = vector.data;
    const data                   = new Uint8Array([...vector].map((x) => x ? 1 : 0));
    return new Column<ArrowToCUDFType<T>>({type: TypeId.BOOL8, data, nullMask});
  }
  visitInt8<T extends Int8>({length, data: {values: data, nullBitmap: nullMask}}: Vector<T>) {
    return new Column<ArrowToCUDFType<T>>({type: TypeId.INT8, length, data, nullMask});
  }
  visitInt16<T extends Int16>({length, data: {values: data, nullBitmap: nullMask}}: Vector<T>) {
    return new Column<ArrowToCUDFType<T>>({type: TypeId.INT16, length, data, nullMask});
  }
  visitInt32<T extends Int32>({length, data: {values: data, nullBitmap: nullMask}}: Vector<T>) {
    return new Column<ArrowToCUDFType<T>>({type: TypeId.INT32, length, data, nullMask});
  }
  visitInt64<T extends Int64>({length, data: {values: data, nullBitmap: nullMask}}: Vector<T>) {
    return new Column<ArrowToCUDFType<T>>({type: TypeId.INT64, length, data, nullMask});
  }
  visitUint8<T extends Uint8>({length, data: {values: data, nullBitmap: nullMask}}: Vector<T>) {
    return new Column<ArrowToCUDFType<T>>({type: TypeId.UINT8, length, data, nullMask});
  }
  visitUint16<T extends Uint16>({length, data: {values: data, nullBitmap: nullMask}}: Vector<T>) {
    return new Column<ArrowToCUDFType<T>>({type: TypeId.UINT16, length, data, nullMask});
  }
  visitUint32<T extends Uint32>({length, data: {values: data, nullBitmap: nullMask}}: Vector<T>) {
    return new Column<ArrowToCUDFType<T>>({type: TypeId.UINT32, length, data, nullMask});
  }
  visitUint64<T extends Uint64>({length, data: {values: data, nullBitmap: nullMask}}: Vector<T>) {
    return new Column<ArrowToCUDFType<T>>({type: TypeId.UINT64, length, data, nullMask});
  }
  // visitFloat16<T extends Float16>(vector: Vector<T>) {}
  visitFloat32<T extends Float32>({length, data: {values: data, nullBitmap: nullMask}}: Vector<T>) {
    return new Column<ArrowToCUDFType<T>>({type: TypeId.FLOAT32, length, data, nullMask});
  }
  visitFloat64<T extends Float64>({length, data: {values: data, nullBitmap: nullMask}}: Vector<T>) {
    return new Column<ArrowToCUDFType<T>>({type: TypeId.FLOAT64, length, data, nullMask});
  }
  visitUtf8<T extends Utf8>({length, data: {values, valueOffsets, nullBitmap}}: Vector<T>) {
    return new Column<ArrowToCUDFType<T>>({
      length,
      type: TypeId.STRING,
      nullMask: nullBitmap,
      children: [
        // offsets
        new Column(
          {type: TypeId.INT32, length: length + 1, data: valueOffsets.subarray(0, length + 1)}),
        // data
        new Column({
          type: TypeId.UINT8,
          length: valueOffsets[length],
          data: values.subarray(0, valueOffsets[length])
        }),
      ]
    });
  }
  // visitBinary<T extends Binary>(vector: Vector<T>) {}
  // visitFixedSizeBinary<T extends FixedSizeBinary>(vector: Vector<T>) {}
  // visitDate<T extends Date_>(vector: Vector<T>) {}
  // visitDateDay<T extends DateDay>(vector: Vector<T>) {}
  // visitDateMillisecond<T extends DateMillisecond>(vector: Vector<T>) {}
  // visitTimestampSecond<T extends TimestampSecond>(vector: Vector<T>) {}
  // visitTimestampMillisecond<T extends TimestampMillisecond>(vector: Vector<T>) {}
  // visitTimestampMicrosecond<T extends TimestampMicrosecond>(vector: Vector<T>) {}
  // visitTimestampNanosecond<T extends TimestampNanosecond>(vector: Vector<T>) {}
  // visitTimeSecond<T extends TimeSecond>(vector: Vector<T>) {}
  // visitTimeMillisecond<T extends TimeMillisecond>(vector: Vector<T>) {}
  // visitTimeMicrosecond<T extends TimeMicrosecond>(vector: Vector<T>) {}
  // visitTimeNanosecond<T extends TimeNanosecond>(vector: Vector<T>) {}
  // visitDecimal<T extends Decimal>(vector: Vector<T>) {}
  visitList<T extends List>(vector: Vector<T>) {
    const {length, data: {valueOffsets, nullBitmap}} = vector;
    return new Column<ArrowToCUDFType<T, {0: ArrowToCUDFType<T['valueType']>}>>({
      length,
      type: TypeId.LIST,
      nullMask: nullBitmap,
      children: [
        // offsets
        new Column(
          {type: TypeId.INT32, length: length + 1, data: valueOffsets.subarray(0, length + 1)}),
        // elements
        this.visit(vector.getChildAt(0) as Vector<T['valueType']>),
      ]
    });
  }
  // visitStruct<T extends Struct>(vector: Vector<T>) {}
  // visitDenseUnion<T extends DenseUnion>(vector: Vector<T>) {}
  // visitSparseUnion<T extends SparseUnion>(vector: Vector<T>) {}
  // visitDictionary<T extends Dictionary>(vector: Vector<T>) {}
  // visitIntervalDayTime<T extends IntervalDayTime>(vector: Vector<T>) {}
  // visitIntervalYearMonth<T extends IntervalYearMonth>(vector: Vector<T>) {}
  // visitFixedSizeList<T extends FixedSizeList>(vector: Vector<T>) {}
  // visitMap<T extends Map_>(vector: Vector<T>) {}
}

const visitor = new VectorToColumnVisitor();

export function fromArrow<T extends DataType>(vector: Vector<T>): Column<ArrowToCUDFType<T>> {
  return visitor.visit(vector);
}
