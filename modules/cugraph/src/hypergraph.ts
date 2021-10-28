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
import {TypeMap} from '@rapidsai/cudf';

import {Graph} from './graph';
import {renumber_edges, renumber_nodes} from './renumber';

export interface HypergraphBaseProps {
  columns?: string[]|null;
  dropNulls?: boolean;
  categories?: {[key: string]: string};
  dropEdgeAttrs?: boolean;
  skip?: string[];
  delim?: string;
  source?: string;
  target?: string;
  nodeId?: string;
  eventId?: string;
  attribId?: string;
  category?: string;
  edgeType?: string;
  nodeType?: string;
}

export interface HypergraphProps extends HypergraphBaseProps {
  attribId?: string;
}

export interface HypergraphDirectProps extends HypergraphBaseProps {
  edgeShape?: {[key: string]: string[]};
  source?: string;
  target?: string;
}

export type HypergraphReturn = {
  nodes: DataFrame,
  edges: DataFrame,
  graph: Graph,
  events: DataFrame,
  entities: DataFrame,
}

export function
hypergraph<T extends TypeMap = any>(values: DataFrame<T>, {
  columns       = null,
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
}: HypergraphProps = {}): HypergraphReturn {
  const computed_columns = _compute_columns(values, columns, skip);

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

  const graph = create_graph(edges, attribId, eventId);

  return {nodes, edges, events, entities, graph};
}

export function hypergraphDirect<T extends TypeMap = any>(values: DataFrame<T>, {
  columns       = null,
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
  const computed_columns = _compute_columns(values, columns, skip);

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

  const graph = create_graph(edges, source, target);

  return {nodes, edges, events, entities, graph};
}

function _compute_columns(values: DataFrame, columns: string[]|null, skip: string[]) {
  const result: string[] = [];
  for (const name of columns ?? values.names) {
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
    col       = dropNulls ? col.dropNulls() : col;  // nansToNulls?
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
                  .dropDuplicates('first', true, true, [nodeId])  // correct defaults?
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

function create_graph(edges: DataFrame, source: string, target: string): Graph {
  const src = edges.get(source);
  const dst = edges.get(target);

  const rnodes = renumber_nodes(src, dst);
  const redges = renumber_edges(src, dst, rnodes);

  return Graph.from_edgelist(redges, {source: 'src', destination: 'dst'});
}

function _prepend_str(series: Series, val: string, delim: string): Series<Utf8String> {
  const prefix = _scalar_init(val, series.length);
  return StringSeries.concatenate([prefix, series.cast(new Utf8String)],
                                  {nullRepr: 'null', separator: delim});
}

function _scalar_init(val: string, size: number): Series<Utf8String> {
  const indices = Series.sequence({init: 0, size: size, step: 0, type: new Int32});
  return Series.new([val]).gather(indices);
}
