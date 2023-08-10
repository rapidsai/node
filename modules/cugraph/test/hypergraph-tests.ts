// Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
import {DataFrame, Int32, Series} from '@rapidsai/cudf';
import {hypergraph, hypergraphDirect} from '@rapidsai/cugraph';
import {CudaMemoryResource, DeviceBuffer} from '@rapidsai/rmm';

const mr = new CudaMemoryResource();

setDefaultAllocator((byteLength: number) => new DeviceBuffer(byteLength, mr));

const simple_df = new DataFrame({
  id: Series.new(['a', 'b', 'c']),
  a1: Series.new([1, 2, 3]).cast(new Int32),
  a2: Series.new(['red', 'blue', 'green']),
  '🙈': Series.new(['æski ēˈmōjē', '😋', 's']),
});

const hyper_df = new DataFrame({
  aa: Series.new([0, 1, 2]).cast(new Int32),
  bb: Series.new(['a', 'b', 'c']),
  cc: Series.new(['b', '0', '1']),
});

test('hyperedges', () => {
  const h = hypergraph(simple_df);
  expect('entities' in h);
  expect(h.entities.numRows).toBe(12);

  expect('nodes' in h);
  expect(h.nodes.numRows).toBe(15);

  expect('edges' in h);
  expect(h.edges.numRows).toBe(12);

  expect('events' in h);
  expect(h.events.numRows).toBe(3);

  expect('graph' in h);

  const edges = h.edges;

  expect([...edges.get('event_id')]).toEqual([
    'event_id::0',
    'event_id::1',
    'event_id::2',
    'event_id::0',
    'event_id::1',
    'event_id::2',
    'event_id::0',
    'event_id::1',
    'event_id::2',
    'event_id::0',
    'event_id::1',
    'event_id::2',
  ]);

  expect([
    ...edges.get('edge_type')
  ]).toEqual(['a1', 'a1', 'a1', 'a2', 'a2', 'a2', 'id', 'id', 'id', '🙈', '🙈', '🙈']);

  expect([...edges.get('attrib_id')]).toEqual([
    'a1::1',
    'a1::2',
    'a1::3',
    'a2::red',
    'a2::blue',
    'a2::green',
    'id::a',
    'id::b',
    'id::c',
    '🙈::æski ēˈmōjē',
    '🙈::😋',
    '🙈::s',
  ]);

  expect([
    ...edges.get('id')
  ]).toEqual(['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c']);

  expect([...edges.get('a1')]).toEqual([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]);

  expect([...edges.get('a2')]).toEqual([
    'red',
    'blue',
    'green',
    'red',
    'blue',
    'green',
    'red',
    'blue',
    'green',
    'red',
    'blue',
    'green'
  ]);
  expect([...edges.get('🙈')]).toEqual([
    'æski ēˈmōjē',
    '😋',
    's',
    'æski ēˈmōjē',
    '😋',
    's',
    'æski ēˈmōjē',
    '😋',
    's',
    'æski ēˈmōjē',
    '😋',
    's'
  ]);
});

test('hyperedges_direct', () => {
  const h = hypergraphDirect(hyper_df);

  expect('edges' in h);
  expect(h.edges.numRows).toBe(9);

  expect('nodes' in h);
  expect(h.nodes.numRows).toBe(9);
});

test('hyperedges_direct_categories', () => {
  const h = hypergraphDirect(
    hyper_df,
    {categories: {aa: 'N', bb: 'N', cc: 'N'}},
  );

  expect('edges' in h);
  expect(h.edges.numRows).toBe(9);

  expect('nodes' in h);
  expect(h.nodes.numRows).toBe(6);
});

test('hyperedges_direct_manual_shaping', () => {
  const h1 = hypergraphDirect(
    hyper_df,
    {edgeShape: {'aa': ['cc'], 'cc': ['cc']}},
  );

  expect('edges' in h1);
  expect(h1.edges.numRows).toBe(6);

  const h2 = hypergraphDirect(
    hyper_df,
    {edgeShape: {'aa': ['cc', 'bb', 'aa'], 'cc': ['cc']}},
  );

  expect('edges' in h2);
  expect(h2.edges.numRows).toBe(12);
});

test('drop_edge_attrs', () => {
  const h = hypergraph(simple_df, {columns: ['id', 'a1', '🙈'], dropEdgeAttrs: true});
  expect('entities' in h);
  expect(h.entities.numRows).toBe(9);

  expect('nodes' in h);
  expect(h.nodes.numRows).toBe(12);

  expect('edges' in h);
  expect(h.edges.numRows).toBe(9);

  expect('events' in h);
  expect(h.events.numRows).toBe(3);

  expect('graph' in h);

  const edges = h.edges;

  expect([...edges.get('event_id')]).toEqual([
    'event_id::0',
    'event_id::1',
    'event_id::2',
    'event_id::0',
    'event_id::1',
    'event_id::2',
    'event_id::0',
    'event_id::1',
    'event_id::2',
  ]);

  expect([
    ...edges.get('edge_type')
  ]).toEqual(['a1', 'a1', 'a1', 'id', 'id', 'id', '🙈', '🙈', '🙈']);

  expect([...edges.get('attrib_id')]).toEqual([
    'a1::1',
    'a1::2',
    'a1::3',
    'id::a',
    'id::b',
    'id::c',
    '🙈::æski ēˈmōjē',
    '🙈::😋',
    '🙈::s',
  ]);
});

test('drop_edge_attrs_direct', () => {
  const h = hypergraphDirect(
    simple_df,
    {columns: ['id', 'a1', '🙈'], edgeShape: {'id': ['a1'], 'a1': ['🙈']}, dropEdgeAttrs: true});
  expect('entities' in h);
  expect(h.entities.numRows).toBe(9);

  expect('nodes' in h);
  expect(h.nodes.numRows).toBe(9);

  expect('edges' in h);
  expect(h.edges.numRows).toBe(6);

  expect('events' in h);
  expect(h.events.numRows).toBe(0);

  expect('graph' in h);

  const edges = h.edges;

  expect([...edges.get('event_id')]).toEqual([
    'event_id::0',
    'event_id::1',
    'event_id::2',
    'event_id::0',
    'event_id::1',
    'event_id::2',
  ]);

  expect([
    ...edges.get('edge_type')
  ]).toEqual(['a1::🙈', 'a1::🙈', 'a1::🙈', 'id::a1', 'id::a1', 'id::a1']);

  expect([...edges.get('src')]).toEqual(['a1::1', 'a1::2', 'a1::3', 'id::a', 'id::b', 'id::c']);
  expect([
    ...edges.get('dst')
  ]).toEqual(['🙈::æski ēˈmōjē', '🙈::😋', '🙈::s', 'a1::1', 'a1::2', 'a1::3']);
});

test('skip_hyper', () => {
  const df = new DataFrame({
    a: Series.new(['a', null, 'b']),
    b: Series.new(['a', 'b', 'c']),
    c: Series.new([1, 2, 3]).cast(new Int32),
  });
  const h  = hypergraph(df, {skip: ['c'], dropNulls: false});
  expect(h.graph.numNodes).toBe(9);
  expect(h.graph.numEdges).toBe(6);
});

test('skip_dropNulls_hyper', () => {
  const df = new DataFrame({
    a: Series.new(['a', null, 'b']),
    b: Series.new(['a', 'b', 'c']),
    c: Series.new([1, 2, 3]).cast(new Int32),
  });
  const h  = hypergraph(df, {skip: ['c'], dropNulls: true});
  expect(h.graph.numNodes).toBe(8);
  expect(h.graph.numEdges).toBe(5);
});

test('skip_direct', () => {
  const df = new DataFrame({
    a: Series.new(['a', null, 'b']),
    b: Series.new(['a', 'b', 'c']),
    c: Series.new([1, 2, 3]).cast(new Int32),
  });
  const h  = hypergraphDirect(df, {skip: ['c'], dropNulls: false});
  expect(h.graph.numNodes).toBe(6);
  expect(h.graph.numEdges).toBe(3);
});

test('skip_dropNulls_direct', () => {
  const df = new DataFrame({
    a: Series.new(['a', null, 'b']),
    b: Series.new(['a', 'b', 'c']),
    c: Series.new([1, 2, 3]).cast(new Int32),
  });
  const h  = hypergraphDirect(df, {skip: ['c'], dropNulls: true});
  expect(h.graph.numNodes).toBe(4);
  expect(h.graph.numEdges).toBe(2);
});

test('dropNulls_hyper', () => {
  const df = new DataFrame({
    a: Series.new(['a', null, 'c']),
    i: Series.new([1, 2, null]).cast(new Int32),
  });
  const h  = hypergraph(df, {dropNulls: true});
  expect(h.graph.numNodes).toBe(7);
  expect(h.graph.numEdges).toBe(4);
});

test('dropNulls_direct', () => {
  const df = new DataFrame({
    a: Series.new(['a', null, 'a']),
    i: Series.new([1, 1, null]).cast(new Int32),
  });
  const h  = hypergraphDirect(df, {dropNulls: true});
  expect(h.graph.numNodes).toBe(2);
  expect(h.graph.numEdges).toBe(1);
});

test('skip_skip_null_hyperedge', () => {
  const df = new DataFrame({
    x: Series.new(['a', 'b', 'c']),
    y: Series.new(['aa', null, 'cc']),
  });

  const expected_hits = ['a', 'b', 'c', 'aa', 'cc'];

  const skip_attr_h_edges = hypergraph(df, {dropEdgeAttrs: true}).edges;

  expect(skip_attr_h_edges.numRows).toBe(expected_hits.length);

  const default_h_edges = hypergraph(df).edges;
  expect(default_h_edges.numRows).toBe(expected_hits.length);
});
