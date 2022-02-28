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

import {CommonType, DataFrame, DataType, Float32, Int32, scope, Series} from '@rapidsai/cudf';

type Nodes<TSource extends DataType, TTarget extends DataType> =
  DataFrame<{id: Int32, node: CommonType<TSource, TTarget>}>;

type Edges<TSource extends DataType, TTarget extends DataType> =
  DataFrame<{idx: Int32, src: TSource, dst: TTarget, weight: Float32}>;

export function renumberNodes<TSource extends DataType, TTarget extends DataType>(
  src: Series<TSource>, dst: Series<TTarget>) {
  return scope(() => {
    const node = src.concat(dst).unique();
    return new DataFrame<{id: Int32, node: CommonType<TSource, TTarget>}>(
             {id: node.encodeLabels(undefined, new Int32), node: node as any})
      .sortValues({id: {ascending: true}});
  }, [src, dst]);
}

export function renumberEdges<TSource extends DataType, TTarget extends DataType>(
  src: Series<TSource>,
  dst: Series<TTarget>,
  weight: Series<Float32>,
  nodes: Nodes<TSource, TTarget>) {
  return scope(() => {
    const idx   = Series.sequence({size: src.length});
    const edges = new DataFrame<{idx: Int32, src: TSource, dst: TTarget, weight: Float32}>(
      {idx, src: src as any, dst: dst as any, weight});
    return renumberTargets(renumberSources(edges, nodes), nodes)
      .sortValues({idx: {ascending: true}})
      .rename({idx: 'id'})
      .select(['id', 'src', 'dst', 'weight']);
  }, [src, dst]);
}

function renumberSources<TSource extends DataType, TTarget extends DataType>(
  edges: Edges<TSource, TTarget>, nodes: Nodes<TSource, TTarget>) {
  return scope(() => {
    const src = edges.assign({node: edges.get('src')})
                  .join({other: nodes, on: ['node']})
                  .sortValues({idx: {ascending: true}})
                  .get('id');
    return edges.assign({src});
  }, [edges, nodes]);
}

function renumberTargets<TSource extends DataType, TTarget extends DataType>(
  edges: Edges<Int32, TTarget>, nodes: Nodes<TSource, TTarget>) {
  return scope(() => {
    const dst = edges.assign({node: edges.get('dst')})
                  .join({other: nodes, on: ['node']})
                  .sortValues({idx: {ascending: true}})
                  .get('id');
    return edges.assign({dst});
  }, [edges, nodes]);
}
