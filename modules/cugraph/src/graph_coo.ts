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

import CuGraph from './addon'

import {Column, Table} from '@nvidia/cudf';
import {CudaMemoryResource} from '@nvidia/rmm';


export interface CuGraphGraphCOOConstructor {
    readonly prototype: CuGraphGraphCOO;
    new(src: Column, dst: Column): CuGraphGraphCOO;
    new(src: Column, dst: Column, stream?: number): CuGraphGraphCOO;
    new(src: Column, dst: Column, stream?: number, mr?: CudaMemoryResource): CuGraphGraphCOO;
}

interface CuGraphGraphCOO {
    readonly numberOfEdges: number;
    readonly numberOfNodes: number;
}

export class GraphCOO extends  (<CuGraphGraphCOOConstructor> CuGraph.GraphCOO)  {
    constructor(data: Table, src_name: string, dst_name: string, stream?: number, mr?: CudaMemoryResource) {
        const src_index = data.columns.indexOf(src_name);
        const dst_index = data.columns.indexOf(dst_name);
        const src = data.getColumn(src_index);
        const dst = data.getColumn(dst_index);
        switch (arguments.length) {
            case 3: super(src, dst); break;
            case 4: super(src, dst, stream); break;
            case 5: super(src, dst, stream, mr); break;
        }
    }
}
