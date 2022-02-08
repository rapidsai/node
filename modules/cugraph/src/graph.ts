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

import {Float32Buffer, MemoryView} from '@rapidsai/cuda';
import {DataFrame, DataType, Int32, scope, Series} from '@rapidsai/cudf';
import {DeviceBuffer} from '@rapidsai/rmm';

import {GraphCOO} from './addon';
import {CUGraph, ForceAtlas2Options} from './node_cugraph';
import {renumberEdges, renumberNodes} from './renumber';

export interface GraphOptions {
  directedEdges?: boolean;
}

export class Graph<T extends DataType = any> {
  public static fromEdgeList<T extends DataType>(src: Series<T>,
                                                 dst: Series<T>,
                                                 options: GraphOptions = {directedEdges: true}) {
    const nodes = renumberNodes(src, dst);
    const edges = renumberEdges(src, dst, nodes);
    return new Graph(nodes, edges, options);
  }

  protected constructor(nodes: DataFrame<{id: Int32, node: T}>,
                        edges: DataFrame<{id: Int32, src: Int32, dst: Int32}>,
                        options: GraphOptions = {directedEdges: true}) {
    this._edges    = edges;
    this._nodes    = nodes;
    this._directed = options?.directedEdges ?? true;
  }

  declare protected _nodes: DataFrame<{id: Int32, node: T}>;
  declare protected _edges: DataFrame<{id: Int32, src: Int32, dst: Int32}>;
  declare protected _directed: boolean;

  declare protected _graph: CUGraph;

  protected get graph(): CUGraph {
    return this._graph || (this._graph = new GraphCOO(this._edges.get('src')._col,
                                                      this._edges.get('dst')._col,
                                                      {directedEdges: this._directed}));
  }

  /**
   * @summary The number of edges in this Graph
   */
  public get numEdges() { return this.graph.numEdges(); }

  /**
   * @summary The number of nodes in this Graph
   */
  public get numNodes() { return this.graph.numNodes(); }

  public get nodes() { return this._nodes.drop(['id']); }

  public get edges() {
    const unnumber = (typ: 'src'|'dst') => {
      const id  = this._edges.get(typ);
      const eid = this._edges.get('id');
      const lhs = new DataFrame({id, eid});
      const rhs = this._nodes.rename({node: typ});
      return lhs.join({on: ['id'], other: rhs});
    };

    return scope(() => unnumber('src')
                         .join({on: ['eid'], other: unnumber('dst')})  //
                         .sortValues({eid: {ascending: true}})
                         .drop(['eid']),
                 [this._edges, this._nodes]);
  }

  public get nodeIds() { return this._nodes.select(['id', 'node']); }

  public get edgeIds() { return this._edges.select(['id', 'src', 'dst']); }

  /**
   * @summary Compute the total number of edges incident to a vertex (both in and out edges).
   */
  public degree() {
    return new DataFrame({id: this._nodes.get('id')._col, degree: this.graph.degree()});
  }

  /**
   * @summary ForceAtlas2 is a continuous graph layout algorithm for handy network visualization.
   *
   * @note Peak memory allocation occurs at 30*V.
   *
   * @param {ForceAtlas2Options} options
   *
   * @returns {Float32Buffer} The new positions.
   */
  public forceAtlas2(options: ForceAtlas2Options = {}) {
    const {numNodes} = this;
    let positions: Float32Buffer|undefined;
    if (options.positions) {
      positions = options.positions ? new Float32Buffer(options.positions instanceof MemoryView
                                                          ? options.positions?.buffer
                                                          : options.positions)
                                    : undefined;
      if (positions && positions.length !== numNodes * 2) {
        // reallocate new positions and copy over old X/Y positions
        const p =
          new Float32Buffer(new DeviceBuffer(numNodes * 2 * Float32Buffer.BYTES_PER_ELEMENT));
        if (positions.length > 0) {
          const pn = positions.length / 2;
          const sx = positions.subarray(0, Math.min(numNodes, pn));
          const sy = positions.subarray(pn, pn + Math.min(numNodes, pn));
          p.copyFrom(sx, 0, 0).copyFrom(sy, 0, numNodes);
        }
        positions = p;
      }
    }
    return new Float32Buffer(this.graph.forceAtlas2({...options, positions}));
  }
}
