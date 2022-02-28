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
import {DataFrame, DataType, Float32, Int32, scope, Series} from '@rapidsai/cudf';
import {DeviceBuffer} from '@rapidsai/rmm';

import {Graph as CUGraph} from './addon';
import {ForceAtlas2Options, SpectralClusteringOptions} from './node_cugraph';
import {renumberEdges, renumberNodes} from './renumber';

export interface GraphOptions {
  directed?: boolean;
}

export class Graph<T extends DataType = any> {
  public static fromEdgeList<T extends DataType>(
    src: Series<T>,
    dst: Series<T>,
    weights = Series.sequence({type: new Float32, size: src.length, init: 1, step: 0}),
    options: GraphOptions = {directed: true}) {
    const nodes = renumberNodes(src, dst);
    const edges = renumberEdges(src, dst, weights, nodes);
    return new Graph(nodes, edges, options);
  }

  protected constructor(nodes: DataFrame<{id: Int32, node: T}>,
                        edges: DataFrame<{id: Int32, src: Int32, dst: Int32, weight: Float32}>,
                        options: GraphOptions = {directed: true}) {
    this._edges    = edges;
    this._nodes    = nodes;
    this._directed = options?.directed ?? true;
  }

  declare protected _nodes: DataFrame<{id: Int32, node: T}>;
  declare protected _edges: DataFrame<{id: Int32, src: Int32, dst: Int32, weight: Float32}>;
  declare protected _directed: boolean;

  declare protected _graph: CUGraph;

  protected get graph() {
    return this._graph || (this._graph = new CUGraph({
                             src: this._edges.get('src')._col,
                             dst: this._edges.get('dst')._col,
                             weight: this._edges.get('weight')._col,
                             directed: this._directed,
                           }));
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
      const lhs = new DataFrame({id, eid, weight: this._edges.get('weight')});
      const rhs = this._nodes.rename({node: typ});
      return lhs.join({on: ['id'], other: rhs});
    };

    return scope(() => unnumber('src')  //
                         .join({on: ['eid'], other: unnumber('dst')})
                         .sortValues({eid: {ascending: true}}),
                 [this])
      .rename({eid: 'id'})
      .select(['id', 'src', 'dst', 'weight']);
  }

  public get nodeIds() { return this._nodes.select(['id']); }

  public get edgeIds() { return this._edges.select(['id', 'src', 'dst']); }

  public dedupeEdges() {
    const src    = this.edges.get('src');
    const dst    = this.edges.get('dst');
    const weight = this.edges.get('weight');
    return DedupedEdgesGraph.fromEdgeList<T>(src, dst, weight, {directed: this._directed});
  }

  /**
   * @summary Compute the total number of edges incident to a vertex (both in and out edges).
   */
  public degree() {
    return new DataFrame({vertex: this._nodes.get('id')._col, degree: this.graph.degree()});
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
    if (options.positions && typeof options.positions === 'object') {
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

export interface ClusteringOptions extends SpectralClusteringOptions {
  type: 'balanced_cut'|'modularity_maximization';
}

export interface AnalyzeClusteringOptions {
  num_clusters: number;
  cluster: Series<Int32>;
  type: 'modularity'|'edge_cut'|'ratio_cut';
}

export class DedupedEdgesGraph<T extends DataType = any> extends Graph<T> {
  public static fromEdgeList<T extends DataType>(
    src: Series<T>,
    dst: Series<T>,
    weights = Series.sequence({type: new Float32, size: src.length, init: 1, step: 0}),
    options: GraphOptions = {directed: true}): DedupedEdgesGraph {
    return scope(() => {
      const ids = new DataFrame({src, dst, id: Series.sequence({size: src.length})})
                    .groupBy({by: ['src', 'dst'], index_key: 'src_dst'})
                    .min();

      const weight = new DataFrame({src, dst, weights: weights.cast(new Float32)})
                       .groupBy({by: ['src', 'dst'], index_key: 'src_dst'})
                       .sum();

      const edges = ids.join({on: ['src_dst'], other: weight}).sortValues({id: {ascending: true}});

      const dd_src = edges.get('src_dst').getChild('src');
      const dd_dst = edges.get('src_dst').getChild('dst');

      const rn_nodes = renumberNodes(dd_src, dd_dst);
      const rn_edges = renumberEdges(dd_src, dd_dst, edges.get('weights'), rn_nodes);

      return new DedupedEdgesGraph(rn_nodes, rn_edges, options);
    }, [src, dst, weights]);
  }

  /**
   * @summary Compute a clustering/partitioning of this graph using either the spectral balanced cut
   * method, or the spectral modularity maximization method.
   *
   * @see https://en.wikipedia.org/wiki/Cluster_analysis
   * @see https://en.wikipedia.org/wiki/Spectral_clustering
   *
   * @param {ClusteringOptions} options Options for the clustering method
   */
  public computeClusters(options: ClusteringOptions) {
    Object.assign(options, {
      num_eigen_vecs: Math.min(2, options.num_clusters),
      evs_tolerance: 0.00001,
      evs_max_iter: 100,
      kmean_tolerance: 0.00001,
      kmean_max_iter: 100,
    });
    const cluster = (() => {
      switch (options.type) {
        case 'balanced_cut': return this.graph.spectralBalancedCutClustering(options);
        case 'modularity_maximization':
          return this.graph.spectralModularityMaximizationClustering(options);
        default: throw new Error(`Unrecognized clustering type "${options.type as string}"`);
      }
    })();
    return new DataFrame({vertex: this._nodes.get('id')._col, cluster});
  }

  /**
   * @summary Compute a score for a given partitioning/clustering. The assumption is
   * that `options.clustering` is the results from a call to {@link computeClusters} and
   * contains columns named `vertex` and `cluster`.
   *
   * @param {AnalyzeClusteringOptions} options
   *
   * @returns {number} The computed clustering score
   */
  public analyzeClustering(options: AnalyzeClusteringOptions) {
    switch (options.type) {
      case 'edge_cut':
        return this.graph.analyzeEdgeCutClustering(options.num_clusters, options.cluster._col);
      case 'ratio_cut':
        return this.graph.analyzeRatioCutClustering(options.num_clusters, options.cluster._col);
      case 'modularity':
        return this.graph.analyzeModularityClustering(options.num_clusters, options.cluster._col);
      default: throw new Error(`Unrecognized clustering type "${options.type as string}"`);
    }
  }
}
