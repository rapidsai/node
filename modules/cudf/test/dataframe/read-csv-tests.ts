// Copyright (c) 2020, NVIDIA CORPORATION.
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

import {setDefaultAllocator} from '@nvidia/cuda';
import {DataFrame, Float64, Int64, String} from '@nvidia/cudf';
import {CudaMemoryResource, DeviceBuffer} from '@nvidia/rmm';

const mr = new CudaMemoryResource();

setDefaultAllocator((byteLength: number) => new DeviceBuffer(byteLength, mr));

describe('DataFrame.readCSV', () => {
  test('can read CSV strings', () => {
    // eslint-disable-next-line @typescript-eslint/ban-types
    const df = DataFrame.readCSV<{a: Int64, b: Float64, c: String}>({
      type: 'buffers',
      sources: [`\
a,b,c
0,1.0,"2"
1,2.0,"3"
2,3.0,"4"\
`].map((v) => Buffer.from(v))
    });
    expect(df.get('a').toArrow().values64).toEqual(new BigInt64Array([0n, 1n, 2n]));
    expect(df.get('b').toArrow().toArray()).toEqual(new Float64Array([1.0, 2.0, 3.0]));
    expect([...df.get('c').toArrow()]).toEqual(["2", "3", "4"]);
  });
});
