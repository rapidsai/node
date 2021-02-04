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

import '@nvidia/rmm';

import {Column, FloatingPoint, Table, Uint32} from '@nvidia/cudf';
import {loadNativeModule} from '@nvidia/rapids-core';

export const CUSPATIAL = loadNativeModule<any>(module, 'node_cuspatial');
export default CUSPATIAL;

export const createQuadtree: <T extends FloatingPoint>(xs: Column<T>,
                                                       ys: Column<T>,
                                                       xMin: number,
                                                       xMax: number,
                                                       yMin: number,
                                                       yMax: number,
                                                       scale: number,
                                                       maxDepth: number,
                                                       minSize: number) => {
  keyMap: Column<Uint32>, table: Table, names: ['key', 'level', 'isQuad', 'length', 'offset']
} = CUSPATIAL.createQuadtree;
