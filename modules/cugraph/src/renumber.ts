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

import {
  CommonType,
  DataFrame,
  DataType,
  Int32,
  Series,
} from '@rapidsai/cudf';

type Nodes<TSource extends DataType, TTarget extends DataType> =
  DataFrame<{id: Int32, node: CommonType<TSource, TTarget>}>;

type Edges<TSource extends DataType, TTarget extends DataType> =
  DataFrame<{src: TSource, dst: TTarget, idx: Int32}>;

export function renumberNodes<TSource extends DataType, TTarget extends DataType>(
  src: Series<TSource>, dst: Series<TTarget>) {
  const tmp1 = src.concat(dst);
  const node = tmp1.unique();
  tmp1.dispose();

  const tmp2 = new DataFrame<{id: Int32, node: CommonType<TSource, TTarget>}>(
    {id: node.encodeLabels(undefined, new Int32), node: node as any});
  const tmp3 = tmp2.sortValues({id: {ascending: true}});
  tmp2.dispose();

  return tmp3;
}

export function renumberEdges<TSource extends DataType, TTarget extends DataType>(
  src: Series<TSource>, dst: Series<TTarget>, nodes: Nodes<TSource, TTarget>) {
  const idx   = Series.sequence({type: new Int32, size: src.length, init: 0});
  const edges = new DataFrame<{src: TSource, dst: TTarget, idx: Int32}>(
    {src: src as any, dst: dst as any, idx});
  const tmp = renumberTargets(renumberSources(edges, nodes), nodes)  //
                .sortValues({idx: {ascending: true}})
                .drop(['idx']);
  idx.dispose();
  return tmp;
}

function renumberSources<TSource extends DataType, TTarget extends DataType>(
  edges: Edges<TSource, TTarget>, nodes: Nodes<TSource, TTarget>) {
  const tmp1 = edges.assign<{node: TSource}>({node: edges.get('src')} as any)
                 .join({other: nodes, on: ['node']});
  const tmp2 = tmp1.sortValues({idx: {ascending: true}});
  tmp1.dispose();

  const tmp3 =
    edges.assign({src: tmp2.get('id')}) as DataFrame<{src: Int32, dst: TTarget, idx: Int32}>;
  tmp2.drop(['id']).dispose();

  return tmp3;
}

function renumberTargets<TSource extends DataType, TTarget extends DataType>(
  edges: Edges<Int32, TTarget>, nodes: Nodes<TSource, TTarget>) {
  const tmp1 = edges.assign<{node: TTarget}>({node: edges.get('dst')} as any)
                 .join({other: nodes, on: ['node']});
  const tmp2 = tmp1.sortValues({idx: {ascending: true}});
  tmp1.dispose();

  const tmp3 =
    edges.assign({dst: tmp2.get('id')}) as DataFrame<{src: Int32, dst: Int32, idx: Int32}>;
  tmp2.drop(['id']).dispose();

  return tmp3;
}
