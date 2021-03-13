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
import {Int32, List, Series} from '@rapidsai/cudf';
import {CudaMemoryResource, DeviceBuffer} from '@rapidsai/rmm';
import * as arrow from 'apache-arrow';
import {VectorType} from 'apache-arrow/interfaces';

const mr = new CudaMemoryResource();

setDefaultAllocator((byteLength) => new DeviceBuffer(byteLength, mr));

describe('ListSeries', () => {
  const validateOffsets = (vec: VectorType<arrow.List>, col: Series<List>) => {
    const expectedOffsets = vec.valueOffsets.subarray(0, vec.length + 1);
    const actualOffsets   = col.offsets.data.toArray();
    expect(expectedOffsets).toEqualTypedArray(actualOffsets);
  };

  const validateElements = (vec: VectorType<arrow.Int32>, col: Series<Int32>) => {
    const expectedElements = vec.values.subarray(0, vec.length);
    const actualElements   = col.data.toArray();
    expect(expectedElements).toEqualTypedArray(actualElements);
  };

  test('Can create from Arrow', () => {
    const vec  = listsOfInt32s([[0, 1, 2], [3, 4, 5]]);
    const ints = vec.getChildAt<arrow.Int32>(0)! as VectorType<arrow.Int32>;
    const col  = Series.new(vec);

    validateOffsets(vec, col);
    validateElements(ints, col.elements);
  });

  test('Can create a List of Lists from Arrow', () => {
    const vec  = listsOfListsOfInt32s([[[0, 1, 2]], [[3, 4, 5], [7, 8, 9]]]);
    const list = vec.getChildAt<ListOfInt32>(0)! as VectorType<ListOfInt32>;
    const ints = list.getChildAt<arrow.Int32>(0)! as VectorType<arrow.Int32>;
    const col  = Series.new(vec);

    validateOffsets(vec, col);
    validateOffsets(list, col.elements);
    validateElements(ints, col.elements.elements);
  });

  test('Can gather a List of Lists', () => {
    const vec  = listsOfListsOfInt32s([[[0, 1, 2]], [[3, 4, 5], [7, 8, 9]]]);
    const list = vec.getChildAt<ListOfInt32>(0)! as VectorType<ListOfInt32>;
    const ints = list.getChildAt<arrow.Int32>(0)! as VectorType<arrow.Int32>;
    const col  = Series.new(vec);
    const out  = col.gather(Series.new({type: new Int32, data: new Int32Array([0, 1, 2])}));

    expect(out.type.children[0].name).toEqual('lists');
    expect(out.type.children[0].type.children[0].name).toEqual('ints');

    validateOffsets(vec, col);
    validateOffsets(list, col.elements);
    validateElements(ints, col.elements.elements);
  });
});

type ListOfInt32 = arrow.List<arrow.Int32>;
type ListOfLists = arrow.List<ListOfInt32>;

function listsOfInt32s(values: number[][]) {
  return arrow.Vector.from({
    values,
    type: new arrow.List(arrow.Field.new({name: 'ints', type: new arrow.Int32})),
  }) as VectorType<ListOfInt32>;
}

function listsOfListsOfInt32s(values: number[][][]) {
  return arrow.Vector.from({
    values,
    type: new arrow.List(arrow.Field.new({
      name: 'lists',
      type: new arrow.List(arrow.Field.new({name: 'ints', type: new arrow.Int32}))
    })),
  }) as VectorType<ListOfLists>;
}
