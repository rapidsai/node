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

/* eslint-disable @typescript-eslint/no-non-null-assertion */

import '../jest-extensions';

import {setDefaultAllocator} from '@nvidia/cuda';
import {Int32, Series} from '@nvidia/cudf';
import {CudaMemoryResource, DeviceBuffer} from '@nvidia/rmm';
import * as arrow from 'apache-arrow';
import {VectorType} from 'apache-arrow/interfaces';

const mr = new CudaMemoryResource();

setDefaultAllocator((byteLength) => new DeviceBuffer(byteLength, mr));

describe('StructSeries', () => {
  const validateElements = (vec: VectorType<arrow.Int32>, col: Series<Int32>) => {
    const expectedElements = vec.values.subarray(0, vec.length);
    const actualElements   = col.data.toArray();
    expect(expectedElements).toEqualTypedArray(actualElements);
  };

  test('Can create from Arrow', () => {
    const vec = structsOfInt32s([
      {x: 0, y: 3},
      {x: 1, y: 4},
      {x: 2, y: 5},
    ]);
    const xs  = vec.getChildAt<arrow.Int32>(0)! as VectorType<arrow.Int32>;
    const ys  = vec.getChildAt<arrow.Int32>(1)! as VectorType<arrow.Int32>;

    const col = Series.new(vec);

    validateElements(xs, col.getChild('x'));
    validateElements(ys, col.getChild('y'));
  });

  test('Can create a Struct of Structs from Arrow', () => {
    const vec    = structsOfStructsOfInt32s([
      {point: {x: 0, y: 3}},
      {point: {x: 1, y: 4}},
      {point: {x: 2, y: 5}},
    ]);
    const points = vec.getChildAt<StructOfInt32>(0)! as VectorType<StructOfInt32>;
    const xs     = points.getChildAt<arrow.Int32>(0)! as VectorType<arrow.Int32>;
    const ys     = points.getChildAt<arrow.Int32>(1)! as VectorType<arrow.Int32>;
    const col    = Series.new(vec);

    validateElements(xs, col.getChild('point').getChild('x'));
    validateElements(ys, col.getChild('point').getChild('y'));
  });
});

type StructOfInt32   = arrow.Struct<{x: arrow.Int32, y: arrow.Int32}>;
type StructOfStructs = arrow.Struct<{point: StructOfInt32}>;

function structsOfInt32s(values: {x: number, y: number}[]) {
  return arrow.Vector.from({
    values,
    type: new arrow.Struct([
      arrow.Field.new({name: 'x', type: new arrow.Int32}),
      arrow.Field.new({name: 'y', type: new arrow.Int32})
    ]),
  }) as VectorType<StructOfInt32>;
}

function structsOfStructsOfInt32s(values: {point: {x: number, y: number}}[]) {
  return arrow.Vector.from({
    values,
    type: new arrow.Struct([arrow.Field.new({
      name: 'point',
      type: new arrow.Struct([
        arrow.Field.new({name: 'x', type: new arrow.Int32}),
        arrow.Field.new({name: 'y', type: new arrow.Int32})
      ]),
    })]),
  }) as VectorType<StructOfStructs>;
}
