// Copyright (c) 2022, NVIDIA CORPORATION.
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
import {DataFrame, Series} from '@rapidsai/cudf';
import {DedupedEdgesGraph} from '@rapidsai/cugraph';
import {CudaMemoryResource, DeviceBuffer} from '@rapidsai/rmm';

const mr = new CudaMemoryResource();

setDefaultAllocator((byteLength: number) => new DeviceBuffer(byteLength, mr));

test.each([
  {type: 'balanced_cut', num_clusters: 2, expected: [1, 0, 0]},
  {type: 'modularity_maximization', num_clusters: 2, expected: [0, 1, 0]},
])(`DedupedEdgesGraph.computeClusters (%j)`, ({type, num_clusters, expected}: any) => {
  const src   = Series.new(new Int32Array([0, 0, 0, 1]));
  const dst   = Series.new(new Int32Array([1, 1, 1, 2]));
  const graph = DedupedEdgesGraph.fromEdgeList(src, dst);
  expect(graph.computeClusters({type, num_clusters}).toString())
    .toEqual(new DataFrame({
               vertex: Series.new(new Int32Array([0, 1, 2])),
               cluster: Series.new(new Int32Array(expected)),
             }).toString());
});

test.each([
  {type: 'edge_cut', clusterType: 'balanced_cut', num_clusters: 2, expected: 1.5},
  {type: 'ratio_cut', clusterType: 'balanced_cut', num_clusters: 2, expected: 3},
  {type: 'modularity', clusterType: 'balanced_cut', num_clusters: 2, expected: -.375},
  {type: 'edge_cut', clusterType: 'modularity_maximization', num_clusters: 2, expected: 2},
  {type: 'ratio_cut', clusterType: 'modularity_maximization', num_clusters: 2, expected: 2.5},
  {type: 'modularity', clusterType: 'modularity_maximization', num_clusters: 2, expected: -.625},
])(`DedupedEdgesGraph.analyzeClustering (%j)`,
   ({type, clusterType, num_clusters, expected}: any) => {
     const src     = Series.new(new Int32Array([0, 0, 0, 1]));
     const dst     = Series.new(new Int32Array([1, 1, 1, 2]));
     const graph   = DedupedEdgesGraph.fromEdgeList(src, dst);
     const cluster = graph.computeClusters({type: clusterType, num_clusters}).get('cluster');
     expect(graph.analyzeClustering({type, num_clusters, cluster})).toEqual(expected);
   });
