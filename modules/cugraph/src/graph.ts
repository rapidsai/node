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

import {DataFrame, DataType, Int32} from '@rapidsai/cudf';

import {GraphCOO} from './addon';
import {renumber_edges, renumber_nodes} from './renumber';

export type EdgelistProps = {
  /** Name of the column holding the source vertices */
  source?: string,
  /** Name of the column holding the destination (target) vertices */
  destination?: string,
}

export class Graph {
  df: DataFrame
  layout: GraphCOO
  renumber_map?: DataFrame<{id: Int32, node: DataType}>

  constructor(df: DataFrame,
              layout: GraphCOO,
              renumber_map?: DataFrame<{id: Int32, node: DataType}>) {
    this.df           = df;
    this.layout       = layout;
    this.renumber_map = renumber_map;
  }

  /**
   * Return a new graph created from the edge list representaion.
   *
   * @param df This DataFrame contains columns storing edge source and destination vetices
   * @param props
   * @returns
   */
  public static from_edgelist(df: DataFrame, {
    source      = 'src',
    destination = 'dst',
  }: EdgelistProps = {}) {
    const src = df.get(source);
    const dst = df.get(destination);

    const renumber_map = renumber_nodes(src, dst);

    const edges = renumber_edges(src, dst, renumber_map);

    const layout =
      new GraphCOO(edges.get('src')._col, edges.get('dst')._col, {directedEdges: true});

    return new Graph(df, layout, renumber_map);
  }
}
