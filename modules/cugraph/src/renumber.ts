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

import {DataFrame, DataType, Float32, Int32, scope, Series} from '@rapidsai/cudf';

type Nodes<TNode extends DataType> = DataFrame<{id: Int32, node: TNode}>;

type Edges<TSource extends DataType, TTarget extends DataType> =
  DataFrame<{idx: Int32, src: TSource, dst: TTarget, weight: Float32}>;

export function renumberNodes<TNode extends DataType>(src: Series<TNode>, dst: Series<TNode>) {
  return scope(() => {
    const node = src.concat(dst).unique();
    const id   = node.encodeLabels(undefined, new Int32);
    return new DataFrame<{id: Int32, node: TNode}>({id, node: node as any})  //
      .sortValues({id: {ascending: true}});
  }, [src, dst]);
}

export function renumberEdges<TNode extends DataType>(
  src: Series<TNode>, dst: Series<TNode>, weight: Series<Float32>, nodes: Nodes<TNode>) {
  return scope(() => {
    const idx   = Series.sequence({size: src.length});
    const edges = new DataFrame<{idx: Int32, src: TNode, dst: TNode, weight: Float32}>(
      {idx, src: src as any, dst: dst as any, weight});
    return renumberTargets(renumberSources(edges, nodes), nodes)
      .sortValues({idx: {ascending: true}})
      .rename({idx: 'id'})
      .select(['id', 'src', 'dst', 'weight']);
  }, [src, dst]);
}

function renumberSources<TSource extends DataType, TTarget extends DataType>(
  edges: Edges<TSource, TTarget>, nodes: Nodes<TSource>) {
  return scope(() => {
    const src = edges.assign({node: edges.get('src')})
                  .join({other: nodes, on: ['node']})
                  .sortValues({idx: {ascending: true}})
                  .get('id');
    return edges.assign({src});
  }, [edges, nodes]);
}

function renumberTargets<TSource extends DataType, TTarget extends DataType>(
  edges: Edges<Int32, TTarget>, nodes: Nodes<TSource>) {
  return scope(() => {
    const dst = edges.assign({node: edges.get('dst')})
                  .join({other: nodes, on: ['node']})
                  .sortValues({idx: {ascending: true}})
                  .get('id');
    return edges.assign({dst});
  }, [edges, nodes]);
}
