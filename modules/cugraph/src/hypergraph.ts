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

import {Categorical, DataFrame, Int32, Series, StringSeries, Utf8String} from '@rapidsai/cudf';
import {TypeMap} from '@rapidsai/cudf';
import {Graph} from './graph';

export type HypergraphBaseProps<T extends TypeMap = any> = {
  /** An optional sequence of column names to process. */
  columns?: readonly(keyof T&string)[],
  /** If True, do not include null values in the graph. */
  dropNulls?: boolean;
  /**
     Dictionary mapping column names to distinct categories. If the same
     value appears columns mapped to the same category, the transform will
     generate one node for it, instead of one for each column.
   */
  categories?: {[key: string]: string};
  /** If True, exclude each row's attributes from its edges (default: False) */
  dropEdgeAttrs?: boolean;
  /** A sequence of column names not to transform into nodes. */
  skip?: readonly(keyof T&string)[],
  /** The delimiter to use when joining column names, categories, and ids. */
  delim?: string;
  /** The name to use as the node id column in the graph and node DataFrame. */
  nodeId?: string;
  /** The name to use as the event id column in the graph and node DataFrames. */
  eventId?: string;
  /** The name to use as the category column in the graph and DataFrames */
  category?: string;
  /** The name to use as the edge type column in the graph and edge DataFrame */
  edgeType?: string;
  /** The name to use as the node type column in the graph and node DataFrames. */
  nodeType?: string;
}

export type HypergraphProps<T extends TypeMap = any> = HypergraphBaseProps<T>&{
  /** The name to use as the category column in the graph and DataFrames */
  attribId?: string;
}

export type HypergraphDirectProps<T extends TypeMap = any> = HypergraphBaseProps<T>&{
  /** Select column pairs instead of making all edges. */
  edgeShape?: {[key: string]: string[]};
  /** The name to use as the source column in the graph and edge DataFrame. */
  source?: string;
  /** The name to use as the target column in the graph and edge DataFrame. */
  target?: string;
}

export type HypergraphReturn = {
  /** A DataFrame of found entity and hyper node attributes. */
  nodes: DataFrame,
  /** A DataFrame of edge attributes. */
  edges: DataFrame,
  /**  Graph of the found entity nodes, hyper nodes, and edges. */
  graph: Graph,
  /** a DataFrame of hyper node attributes for direct graphs, else empty. */
  events: DataFrame,
  /** A DataFrame of the found entity node attributes. */
  entities: DataFrame,
}

/**
 * Creates a hypergraph out of the given dataframe, returning the graph components as dataframes.
 *
 * The transform reveals relationships between the rows and unique values. This transform is useful
 * for lists of events, samples, relationships, and other structured high-dimensional data.  The
 * transform creates a node for every row, and turns a row's column entries into node attributes.
 * Every unique value within a column is also turned into a node.
 *
 * Edges are added to connect a row's nodes to each of its column nodes. Nodes are given the
 * attribute specified by ``nodeType`` that corresponds to the originating column name, or if a row
 * ``eventId``.
 *
 * Consider a list of events. Each row represents a distinct event, and each column some metadata
 * about an event. If multiple events have common metadata, they will be transitively connected
 * through those metadata values. Conversely, if an event has unique metadata, the unique metadata
 * will turn into nodes that only have connections to the event node.  For best results, set
 * ``eventId`` to a row's unique ID, ``skip`` to all non-categorical columns (or ``columns`` to all
 * categorical columns), and ``categories`` to group columns with the same kinds of values.
 */
export function
hypergraph<T extends TypeMap = any>(values: DataFrame<T>, {
  columns       = values.names,
  dropNulls     = true,
  categories    = {},
  dropEdgeAttrs = false,
  skip          = [],
  delim         = '::',
  nodeId        = 'node_id',
  eventId       = 'event_id',
  attribId      = 'attrib_id',
  category      = 'category',
  edgeType      = 'edge_type',
  nodeType      = 'node_type',
}: HypergraphProps<T> = {}): HypergraphReturn {
  const computed_columns = _compute_columns(columns, skip);

  const initial_events =
    _create_events(values, computed_columns, delim, dropNulls, eventId, nodeType);

  const entities = _create_entity_nodes(
    initial_events, computed_columns, dropNulls, categories, delim, nodeId, category, nodeType);

  const edges  = _create_hyper_edges(initial_events,
                                    computed_columns,
                                    dropNulls,
                                    categories,
                                    dropEdgeAttrs,
                                    delim,
                                    eventId,
                                    attribId,
                                    category,
                                    edgeType,
                                    nodeType);
  const events = _create_hyper_nodes(initial_events, nodeId, eventId, category, nodeType);
  const nodes  = entities.concat(events);

  const graph = Graph.fromEdgeList(edges.get(attribId), edges.get(eventId));

  return {nodes, edges, events, entities, graph};
}

/**
 * Creates a hypergraph out of the given dataframe, returning the graph components as dataframes.
 *
 * The transform reveals relationships between the rows and unique values. This transform is useful
 * for lists of events, samples, relationships, and other structured high-dimensional data.  The
 * transform creates a node for every row, and turns a row's column entries into node attributes.
 * Every unique value within a column is also turned into a node.
 *
 * Edges are added to connect a row's nodes to one another. Nodes are given the attribute specified
 * by ``nodeType`` that corresponds to the originating column name, or if a row ``eventId``.
 *
 * Consider a list of events. Each row represents a distinct event, and each column some metadata
 * about an event. If multiple events have common metadata, they will be transitively connected
 * through those metadata values. Conversely, if an event has unique metadata, the unique metadata
 * will turn into nodes that only have connections to the event node.  For best results, set
 * ``eventId`` to a row's unique ID, ``skip`` to all non-categorical columns (or ``columns`` to all
 * categorical columns), and ``categories`` to group columns with the same kinds of values.
 */
export function hypergraphDirect<T extends TypeMap = any>(values: DataFrame<T>, {
  columns       = values.names,
  categories    = {},
  dropNulls     = true,
  edgeShape     = {},
  dropEdgeAttrs = false,
  skip          = [],
  delim         = '::',
  source        = 'src',
  target        = 'dst',
  nodeId        = 'node_id',
  eventId       = 'event_id',
  category      = 'category',
  edgeType      = 'edge_type',
  nodeType      = 'node_type',
}: HypergraphDirectProps = {}): HypergraphReturn {
  const computed_columns = _compute_columns(columns, skip);

  const initial_events =
    _create_events(values, computed_columns, delim, dropNulls, eventId, nodeType);

  const entities = _create_entity_nodes(
    initial_events, computed_columns, dropNulls, categories, delim, nodeId, category, nodeType);

  const edges  = _create_direct_edges(initial_events,
                                     computed_columns,
                                     dropNulls,
                                     categories,
                                     edgeShape,
                                     dropEdgeAttrs,
                                     delim,
                                     source,
                                     target,
                                     eventId,
                                     category,
                                     edgeType,
                                     nodeType);
  const events = new DataFrame({});
  const nodes  = entities;

  const graph = Graph.fromEdgeList(edges.get(source), edges.get(target));

  return {nodes, edges, events, entities, graph};
}

function _compute_columns<T extends TypeMap = any>(columns: readonly(keyof T & string)[],
                                                   skip: readonly(keyof T & string)[]) {
  const result: string[] = [];
  for (const name of columns) {
    if (!skip.includes(name)) { result.push(name); }
  }
  result.sort();
  return result;
}

function _create_events(values: DataFrame,
                        columns: string[],
                        delim: string,
                        dropNulls: boolean,
                        eventId: string,
                        nodeType: string) {
  const series_map: {[key: string]: any} = {};
  for (const name of columns) { series_map[name] = values.get(name); }

  if (!(eventId in series_map)) {
    series_map[eventId] =
      Series.sequence({type: new Int32, init: 0, step: 1, size: values.numRows});
  }

  series_map[eventId]  = _prepend_str(series_map[eventId], eventId, delim);
  series_map[nodeType] = _scalar_init('event', series_map[eventId].length);

  if (!dropNulls) {
    for (const name of columns) {
      const col = series_map[name];
      if (col instanceof StringSeries) { series_map[name] = col.replaceNulls('null'); }
    }
  }

  return new DataFrame(series_map);
}

function _create_entity_nodes(events: DataFrame,
                              columns: string[],
                              dropNulls: boolean,
                              categories: {[key: string]: string},
                              delim: string,
                              nodeId: string,
                              category: string,
                              nodeType: string) {
  const node_dfs: DataFrame[] = [];

  for (const name of columns) {
    const cat = name in categories ? categories[name] : name;
    let col   = events.get(name);
    col       = col.unique();
    col       = dropNulls ? col.dropNulls() : col;
    if (col.length == 0) { continue; }

    const df = new DataFrame({
      [name]: col,
      [nodeId]: _prepend_str(col, cat, delim),
      [category]: _scalar_init(cat, col.length),
      [nodeType]: _scalar_init(name, col.length),
    });
    node_dfs.push(df);
  }
  const nodes = new DataFrame()
                  .concat(...node_dfs)
                  .dropDuplicates('first', true, [nodeId])
                  .select(columns.concat([nodeId, nodeType, category]));

  return nodes;
}

function _create_hyper_nodes(
  events: DataFrame, nodeId: string, eventId: string, category: string, nodeType: string) {
  const series_map: {[key: string]: any} = {};
  for (const name of events.names) { series_map[name] = events.get(name); }

  series_map[nodeType] = _scalar_init(eventId, events.numRows);
  series_map[category] = _scalar_init('event', events.numRows);
  series_map[nodeId]   = series_map[eventId];

  return new DataFrame(series_map);
}

function _create_hyper_edges(events: DataFrame,
                             columns: string[],
                             dropNulls: boolean,
                             categories: {[key: string]: string},
                             dropEdgeAttrs: boolean,
                             delim: string,
                             eventId: string,
                             attribId: string,
                             category: string,
                             edgeType: string,
                             nodeType: string) {
  const edge_attrs = events.names.filter(name => name != nodeType);

  const edge_dfs: DataFrame[] = [];

  for (const name of columns) {
    const cat = name in categories ? categories[name] : name;

    const fs = dropEdgeAttrs ? [eventId, name] : [eventId, ...edge_attrs];

    let df: DataFrame = dropNulls ? events.select(fs).dropNulls(0, 1, [name]) : events.select(fs);

    if (df.numRows == 0) { continue; }

    df = df.assign({
      [edgeType]: _scalar_init(cat, df.numRows),
      [attribId]: _prepend_str(df.get(name), cat, delim),
    });

    if (Object.keys(categories).length > 0) {
      df = df.assign({[category]: _scalar_init(name, df.numRows)});
    }

    edge_dfs.push(df);
  }

  const cols = [eventId, edgeType, attribId];
  if (Object.keys(categories).length > 0) { cols.push(category); }
  if (!dropEdgeAttrs) { cols.push(...edge_attrs); }

  return new DataFrame().concat(...edge_dfs).select(cols);
}

function _create_direct_edges(events: DataFrame,
                              columns: string[],
                              dropNulls: boolean,
                              categories: {[key: string]: string},
                              edgeShape: {[key: string]: string[]},
                              dropEdgeAttrs: boolean,
                              delim: string,
                              source: string,
                              target: string,
                              eventId: string,
                              category: string,
                              edgeType: string,
                              nodeType: string) {
  if (Object.keys(edgeShape).length == 0) {
    columns.forEach((value, index) => edgeShape[value] = columns.slice(index + 1));
  }

  const edge_attrs = events.names.filter(name => name != nodeType);

  const edge_dfs: DataFrame[] = [];

  for (const key1 of Object.keys(edgeShape).sort()) {
    const cat1 = key1 in categories ? categories[key1] : key1;

    for (const key2 of edgeShape[key1].sort()) {
      const cat2 = key2 in categories ? categories[key2] : key2;

      const fs = dropEdgeAttrs ? [eventId, key1, key2] : [eventId, ...edge_attrs];

      let df: DataFrame =
        dropNulls ? events.select(fs).dropNulls(0, 2, [key1, key2]) : events.select(fs);

      if (df.numRows == 0) { continue; }

      if (Object.keys(categories).length > 0) {
        df = df.assign({[category]: _scalar_init(key1 + delim + key2, df.numRows)});
      }

      df = df.assign({
        [edgeType]: _scalar_init(cat1 + delim + cat2, df.numRows),
        [source]: _prepend_str(df.get(key1), cat1, delim),
        [target]: _prepend_str(df.get(key2), cat2, delim),
      });

      edge_dfs.push(df);
    }
  }

  const cols = [eventId, edgeType, source, target];
  if (Object.keys(categories).length > 0) { cols.push(category); }
  if (!dropEdgeAttrs) { cols.push(...edge_attrs); }

  return new DataFrame().concat(...edge_dfs).select(cols);
}

function _prepend_str(series: Series, val: string, delim: string) {
  const prefix = val + delim;
  const suffix = series.cast(new Categorical(new Utf8String));
  const codes  = suffix.codes;
  const categories =
    Series.new(suffix.categories.replaceNulls('null')._col.replaceSlice(prefix, 0, 0));

  return Series.new({type: suffix.type, length: codes.length, children: [codes, categories]});
}

function _scalar_init(val: string, size: number): Series<Utf8String> {
  return Series.new([val]).gather(Series.sequence({size, step: 0}), false);
}
