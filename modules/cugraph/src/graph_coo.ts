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

import { Column } from '@nvidia/cudf';

export interface CuGraphGraphCOOConstructor {
    readonly prototype: CuGraphGraphCOO;
    new(): CuGraphGraphCOO;
}

class Edgelist{
    readonly src: Column;
    readonly dst: Column;

    constructor(src: Column, dst: Column) {
        this.src = src;
        this.dst = dst;
    }
}

interface CuGraphGraphCOO {
    readonly numberOfEdges: number;
    readonly numberOfNodes: number;
    from_edge_list(src: Column, dst: Column): void;    
    clear(): void;
}

export class GraphCOO extends  (<CuGraphGraphCOOConstructor> CuGraph.GraphCOO)  {
    from_edge_list(src: Column, dst: Column): void {
        this.edgelist = new Edgelist(src, dst)
    }

    clear(): void {
        delete this.edgelist;
    }

    edgelist?: Edgelist;
}
