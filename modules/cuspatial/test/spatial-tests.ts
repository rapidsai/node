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

import '@rapidsai/cudf/test/jest-extensions';

import {setDefaultAllocator} from '@rapidsai/cuda';
import {Series} from '@rapidsai/cudf';
import {convertLonLatToCartesian} from '@rapidsai/cuspatial';
import {DeviceBuffer} from '@rapidsai/rmm';

setDefaultAllocator((byteLength: number) => new DeviceBuffer(byteLength));

describe('Spatial', () => {
  test(`convertLonLatToCartesian`, () => {
    const seriesX = Series.new([-120.0, -120.0]);
    const seriesY = Series.new([48.0, 49.0]);
    const result  = convertLonLatToCartesian(1.0, 1.0, seriesX._col, seriesY._col);

    expect(result.y.toArray()).toMatchObject({'0': -5222.222222222223, '1': -5333.333333333334});
    expect(result.x.toArray()).toMatchObject({'0': 12233.92375289575, '1': 12184.804692381627});
  });
});
