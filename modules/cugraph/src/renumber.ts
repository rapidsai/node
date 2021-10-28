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

import {DataFrame, Int32, Series, StringSeries, Utf8String} from '@rapidsai/cudf';

function join(edges: DataFrame, nodes: DataFrame<{id: Int32, node: Utf8String}>, col: string) {
  const tmp = edges.assign({node: edges.get(col)}).join({other: nodes, on: ['node']}).sortValues({
    idx: {ascending: true}
  });
  return edges.drop([col]).assign({[col]: tmp.get('id')});
}

export function renumber_nodes(src: StringSeries, dst: StringSeries) {
  const nodes = src.concat(dst).unique() as StringSeries;
  const ids   = nodes.encodeLabels();
  return new DataFrame({'id': ids, 'node': nodes}).sortValues({id: {ascending: true}});
}

export function renumber_edges(
  src: StringSeries, dst: StringSeries, nodes: DataFrame<{id: Int32, node: Utf8String}>) {
  const idx   = Series.sequence({type: new Int32, size: src.length, init: 0});
  const edges = new DataFrame({src: src, dst: dst, idx: idx});
  return join(join(edges, nodes, 'src'), nodes, 'dst').sortValues({idx: {ascending: true}}).drop([
    'idx'
  ]);
}
