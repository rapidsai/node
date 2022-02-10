// Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

import {setDefaultAllocator} from '@rapidsai/cuda';
import {DataFrame, Float32, Int32, Series} from '@rapidsai/cudf';
import {Graph} from '@rapidsai/cugraph';
import {CudaMemoryResource, DeviceBuffer} from '@rapidsai/rmm';

const mr = new CudaMemoryResource();

setDefaultAllocator((byteLength: number) => new DeviceBuffer(byteLength, mr));

test(`Graph.fromEdgeList`, () => {
  const src   = Series.new(['192.168.1.1', '172.217.5.238', '216.228.121.209', '192.16.31.23']);
  const dst   = Series.new(['172.217.5.238', '216.228.121.209', '192.16.31.23', '192.168.1.1']);
  const graph = Graph.fromEdgeList(src, dst);

  expect(graph.nodes.toString()).toEqual(new DataFrame({
                                           node: src.concat(dst).unique()
                                         }).toString());

  expect(graph.edges.toString()).toEqual(new DataFrame({
                                           id: Series.sequence({size: 4}),
                                           src,
                                           dst,
                                           weight: Series.sequence(
                                             {type: new Float32, size: 4, init: 1, step: 0}),
                                         }).toString());

  expect(graph.nodeIds.toString()).toEqual(new DataFrame({
                                             id: Series.sequence({size: 4})
                                           }).toString());

  expect(graph.edgeIds.toString()).toEqual(new DataFrame({
                                             id: Series.sequence({size: 4}),
                                             src: Series.new({type: new Int32, data: [2, 0, 3, 1]}),
                                             dst: Series.new({type: new Int32, data: [0, 3, 1, 2]}),
                                           }).toString());
});

test(`Graph.degree`, () => {
  const src   = Series.new(['192.168.1.1', '172.217.5.238', '216.228.121.209', '192.16.31.23']);
  const dst   = Series.new(['172.217.5.238', '216.228.121.209', '192.16.31.23', '192.168.1.1']);
  const graph = Graph.fromEdgeList(src, dst);
  expect(graph.degree().toString()).toEqual(new DataFrame({
                                              vertex: Series.sequence({size: 4}),
                                              degree: Series.sequence({size: 4, init: 2, step: 0}),
                                            }).toString());
});
