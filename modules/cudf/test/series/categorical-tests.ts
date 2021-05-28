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

import '../jest-extensions';

import {setDefaultAllocator} from '@nvidia/cuda';
import {Categorical, Series} from '@rapidsai/cudf';
import {CudaMemoryResource, DeviceBuffer} from '@rapidsai/rmm';
import {Data, Dictionary, Int32, Utf8, Vector} from 'apache-arrow';

const mr = new CudaMemoryResource();

setDefaultAllocator((byteLength) => new DeviceBuffer(byteLength, mr));

describe('CategoricalSeries', () => {
  test('Constructs CategoricalSeries from Arrow Dictionary Vector', () => {
    const codes      = Vector.from({type: new Int32(), values: [0, 1, 2, 2, 1, 0]});
    const categories = Vector.from({type: new Utf8(), values: ['foo', 'bar', 'baz']});
    const dictionary = Vector.new(Data.Dictionary(
      new Dictionary(categories.type, codes.type), 0, codes.length, 0, null, codes, categories));

    const categorical = Series.new(dictionary);

    expect([...categorical]).toEqual([...dictionary]);
    expect([...categorical.codes]).toEqual([...codes]);
    expect([...categorical.categories]).toEqual([...categories]);
  });

  test('Can cast a String Series to Categorical', () => {
    const categories  = Series.new(['foo', 'foo', 'bar', 'bar']);
    const categorical = categories.cast(new Categorical(categories.type));
    expect([...categorical]).toEqual(['foo', 'foo', 'bar', 'bar']);
    expect([...categorical.codes]).toEqual([1, 1, 0, 0]);
    expect([...categorical.categories]).toEqual(['bar', 'foo']);
  });
});
