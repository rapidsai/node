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
import {Series} from '@rapidsai/cudf';
import {DedupedEdgesGraph, Graph} from '@rapidsai/cugraph';
import {CudaMemoryResource, DeviceBuffer} from '@rapidsai/rmm';

const mr = new CudaMemoryResource();

setDefaultAllocator((byteLength: number) => new DeviceBuffer(byteLength, mr));

describe('Graph', () => {
  const src   = Series.new(['192.168.1.1', '172.217.5.238', '216.228.121.209', '192.16.31.23']);
  const dst   = Series.new(['172.217.5.238', '216.228.121.209', '192.16.31.23', '192.168.1.1']);
  const graph = Graph.fromEdgeList(src, dst);

  test('numNodes', () => {expect(graph.numNodes).toEqual(4);});
  test('numEdges', () => {expect(graph.numNodes).toEqual(4);});

  test(`nodes`, () => {
    const nodes    = graph.nodes;
    const expected = src.concat(dst).unique();
    expect([...nodes.get('node')]).toEqual([...expected]);
  });

  test(`edges`, () => {
    const edges = graph.edges;
    expect([...edges.get('id')]).toEqual([0, 1, 2, 3]);
    expect([...edges.get('src')]).toEqual([...src]);
    expect([...edges.get('dst')]).toEqual([...dst]);

    expect([...edges.get('weight')]).toEqual([1.0, 1.0, 1.0, 1.0]);
  });

  test(`nodeIds`, () => {
    const ids = graph.nodeIds.get('id');
    expect([...ids]).toEqual([0, 1, 2, 3]);
  });

  test(`edgeIds`, () => {
    const edgeIds = graph.edgeIds;
    expect([...edgeIds.get('id')]).toEqual([0, 1, 2, 3]);
    expect([...edgeIds.get('src')]).toEqual([2, 0, 3, 1]);
    expect([...edgeIds.get('dst')]).toEqual([0, 3, 1, 2]);
  });

  test(`degree`, () => {
    const degree = graph.degree();
    expect([...degree.get('vertex')]).toEqual([0, 1, 2, 3]);
    expect([...degree.get('degree')]).toEqual([2, 2, 2, 2]);
  });

  test(`dedupeEdges`, () => {
    const src   = Series.new(['1', '2', '2', '3', '4', '1']);
    const dst   = Series.new(['2', '3', '3', '4', '1', '2']);
    const graph = Graph.fromEdgeList(src, dst);
    expect(graph.numNodes).toBe(4);
    expect(graph.numEdges).toBe(6);
    const dd_graph = graph.dedupeEdges();
    expect(dd_graph).toBeInstanceOf(DedupedEdgesGraph);
    expect(dd_graph.numNodes).toBe(4);
    expect(dd_graph.numEdges).toBe(4);
  });
});

describe('DedupedEdgesGraph', () => {
  const src    = Series.new(['1', '2', '2', '3', '4', '1']);
  const dst    = Series.new(['2', '3', '3', '4', '1', '2']);
  const dd_src = Series.new(['1', '2', '3', '4']);
  const dd_dst = Series.new(['2', '3', '4', '1']);
  const graph  = DedupedEdgesGraph.fromEdgeList(src, dst);

  test('numNodes', () => {expect(graph.numNodes).toEqual(4);});
  test('numEdges', () => {expect(graph.numNodes).toEqual(4);});

  test(`nodes`, () => {
    const nodes    = graph.nodes;
    const expected = dd_src.concat(dd_dst).unique();
    expect([...nodes.get('node')]).toEqual([...expected]);
  });

  test(`edges`, () => {
    const edges = graph.edges;
    expect([...edges.get('id')]).toEqual([0, 1, 2, 3]);
    expect([...edges.get('src')]).toEqual([...dd_src]);
    expect([...edges.get('dst')]).toEqual([...dd_dst]);

    expect([...edges.get('weight')]).toEqual([2.0, 2.0, 1.0, 1.0]);
  });

  test(`nodeIds`, () => {
    const ids = graph.nodeIds.get('id');
    expect([...ids]).toEqual([0, 1, 2, 3]);
  });

  test(`edgeIds`, () => {
    const edgeIds = graph.edgeIds;
    expect([...edgeIds.get('id')]).toEqual([0, 1, 2, 3]);
    expect([...edgeIds.get('src')]).toEqual([0, 1, 2, 3]);
    expect([...edgeIds.get('dst')]).toEqual([1, 2, 3, 0]);
  });

  test(`degree`, () => {
    const degree = graph.degree();
    expect([...degree.get('vertex')]).toEqual([0, 1, 2, 3]);
    expect([...degree.get('degree')]).toEqual([2, 2, 2, 2]);
  });
});
