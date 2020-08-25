# Copyright (c) 2020, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import cudf
import numpy as np
import pandas as pd
from math import ceil
from cugraph.structure import graph_new_wrapper


def shape_graph(graph=None,
                nodes=None,
                symmetrize=False,
                SOURCE="src",
                TARGET="dst",
                NODEID="node_id",
                CATEGORY="category",
                **kwargs):
    if symmetrize:
        graph = graph.to_directed()
    edges = graph.edges()[[SOURCE, TARGET]]
    nodes = _filter_by(nodes, graph.nodes(), NODEID)
    nid, src, dst = _combine_categories(
        nodes[NODEID], edges[SOURCE], edges[TARGET]
    )
    edge, bundle = _compute_edge_bundles(src, dst)
    nlist = cudf.DataFrame({
        "node": nodes[NODEID],
        "category": nodes[CATEGORY],
        "id": nid.astype(np.uint32),
        "color": _category_to_color(nodes[CATEGORY]),
        "size": _degrees_to_size(graph).astype(np.uint8)
    }).sort_values(by="id", ignore_index=True)
    elist = cudf.DataFrame({
        "src": edges[SOURCE],
        "dst": edges[TARGET],
        "edge": edge.astype(np.uint64),
        "bundle": bundle.astype(np.uint64),
        "color": _stack_columns(cudf.DataFrame({
            "src": _filter_by(nlist[["id", "color"]], src, left_on="id")["color"],
            "dst": _filter_by(nlist[["id", "color"]], dst, left_on="id")["color"],
        }))
    })
    return (graph, nlist, elist)


def _category_to_color(types):
    # from random import shuffle
    color_palette = [
        -12451426,-11583787,-12358156,-10375427,
        -7610114,-4194305,-6752794,-5972565,
        -5914010,-4356046,-6140066
    ]
    # shuffle(color_palette)
    types = types.astype("category")
    color_indices = cudf.Series(types.cat.codes)
    color_palette = cudf.Series(color_palette)
    color_palettes = []
    num_color_ids = color_indices.max() + 1
    for i in range(ceil(num_color_ids / len(color_palette))):
        color_palettes.append(color_palette)
    return cudf.Series(cudf.core.column.build_categorical_column(
        ordered=True,
        codes=color_indices._column,
        categories=cudf.concat(color_palettes)[:num_color_ids],
    ).as_numerical_column(dtype=np.uint32))


def _combine_categories(*cols):
    cols = [col.astype("category") for col in cols]
    cats = cudf.concat([col.cat.categories for col in cols]) \
        .to_series().drop_duplicates(ignore_index=True)._column
    cols = [
        col.cat._set_categories(col.cat.categories, cats, is_unique=True)
        for col in cols
    ]
    dtype = np.uint32 # np.find_common_type([col.cat().codes.dtype for col in cols], [])
    return [col.cat().codes.astype(dtype) for col in cols]


def _compute_edge_bundles(src, dst):
    edges = cudf.DataFrame({"src": src, "dst": dst})
    edges = edges.reset_index().rename({"index": "eid"}, axis=1, copy=False)
    # Create a duplicate table with:
    # * all the [src, dst] in the upper half
    # * all the [dst, src] pairs as the lower half, but flipped so dst->src, src->dst
    bundles = cudf.DataFrame({
        "eid": cudf.concat([edges["eid"], edges["eid"]]),
        # concat [src, dst] into the "src" column
        "src": cudf.concat([edges["src"], edges["dst"]]),
        # concat [dst, src] into the "dst" column
        "dst": cudf.concat([edges["dst"], edges["src"]]),
    })

    # Group the duplicated edgelist by [src, dst] and get the min edge id.
    # Since all the [dst, src] pairs have been flipped to [src, dst], each
    # edge with the same [src, dst] or [dst, src] vertices will be assigned
    # the same bundle id
    bundles = bundles \
        .groupby(["src", "dst"]).agg({"eid": "min"}) \
        .reset_index().rename({"eid": "bid"}, axis=1, copy=False)

    # Join the bundle ids into the edgelist
    edges = edges.merge(bundles, on=["src", "dst"], how="inner")

    # Determine each bundle"s size and relative offset
    bundles = edges["bid"].sort_values()
    lengths = bundles.value_counts(sort=False)
    offsets = lengths.cumsum() - lengths
    # Join the bundle segment lengths + offsets into the edgelist
    edges = edges.merge(cudf.DataFrame({
        "bid": bundles.unique().reset_index(drop=True),
        "start": offsets.reset_index(drop=True).astype(np.uint32),
        "count": lengths.reset_index(drop=True).astype(np.uint32),
    }), on="bid", how="left")

    # Determine each edge"s index relative to its bundle
    edges = edges.sort_values(by="bid").reset_index(drop=True)
    edges["index"] = cudf.core.index.RangeIndex(0, len(edges)) - edges["start"]
    edges["index"] = edges["index"].astype(np.uint32)

    # Re-sort the edgelist by edge id and cleanup
    edges = edges.sort_values("eid").reset_index(drop=True)
    edges = edges.rename({"eid": "id"}, axis=1, copy=False)
    edges = edges[["id", "src", "dst", "index", "count"]]

    return (
        _stack_columns(edges[["src", "dst"]]),
        _stack_columns(edges[["index", "count"]])
    )


def _assign(df1, df2):
    for key, col in df2.iteritems():
        df1[key] = col
    return df1

def _rename(df, rename_map):
    return df.rename(rename_map, axis=1, copy=False)


def _reset_index(df):
    df.reset_index(drop=True, inplace=True)
    return df


def _filter_by(df, sr, left_on=None, right_on=None, out_path=None):
    left_on = str(sr.name if left_on is None else left_on)
    right_on = str(sr.name if right_on is None else right_on)
    sr = sr.to_frame(right_on).set_index(right_on, drop=True)
    df = df.drop_duplicates(left_on).set_index(left_on, drop=False)
    return _reset_index(sr.join(df, how="left"))


def _stack_columns(df):
    size = sum([df[x].dtype.itemsize for x in df])
    data = _reset_index(_reset_index(df).stack()).data
    dtype = cudf.utils.dtypes.min_unsigned_type(0, size*8)
    return cudf.core.column.NumericalColumn(data, dtype=dtype)


def _degrees_to_size(G):
    degrees = graph_new_wrapper._degrees(G)
    degrees = cudf.Series(degrees[1], dtype=np.uint32) + \
              cudf.Series(degrees[2], dtype=np.uint32)
    return degrees.scale() * 250 + 5

# def write_arrow(df, path):
#     import pyarrow as pa
#     table = df.to_arrow()
#     writer = pa.RecordBatchStreamWriter(path, table.schema)
#     writer.write_table(table)
#     writer.close()
