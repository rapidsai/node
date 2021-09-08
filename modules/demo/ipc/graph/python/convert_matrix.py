# Copyright (c) 2021, NVIDIA CORPORATION.
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
import cugraph
import numpy as np
from cugraph.structure.graph_implementation import simpleGraphImpl

EdgeList = simpleGraphImpl.EdgeList

def from_cudf_edgelist(df, source="src", target="dst"):

    """
    Construct an enhanced graph from a cuDF edgelist that doesn't collapse
    duplicate edges and includes columns for node degree and edge bundle.
    """

    def drop_index(df):
        return df.reset_index(drop=True)

    def arange(size, dtype="uint32"):
        return cudf.core.index.RangeIndex(0, size).to_series().astype(dtype)

    def smoosh(df):
        size = sum([df[x].dtype.itemsize for x in df])
        data = drop_index(drop_index(df).stack()).data
        dtype = cudf.utils.dtypes.min_unsigned_type(0, size*8)
        return cudf.core.column.NumericalColumn(data, dtype=dtype)

    def compute_edge_bundles(edges, id_, src, dst):

        edges = cudf.DataFrame({
            "eid": drop_index(edges[id_]),
            "src": drop_index(edges[src]),
            "dst": drop_index(edges[dst]),
        })
        # Create a duplicate table with:
        # * all the [src, dst] in the upper half
        # * all the [dst, src] pairs as the lower half, but flipped so dst->src, src->dst
        bundles = drop_index(cudf.DataFrame({
            "eid": cudf.concat([edges["eid"], edges["eid"]], ignore_index=True),
            # concat [src, dst] into the "src" column
            "src": cudf.concat([edges["src"], edges["dst"]], ignore_index=True),
            # concat [dst, src] into the "dst" column
            "dst": cudf.concat([edges["dst"], edges["src"]], ignore_index=True),
        }))

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
        lengths = edges["bid"].value_counts(sort=False).sort_index()
        bundles = lengths.index.to_series().unique()
        offsets = lengths.cumsum() - lengths

        # Join the bundle segment lengths + offsets into the edgelist
        edges = edges.merge(cudf.DataFrame({
            "bid": drop_index(bundles.astype(np.uint32)),
            "start": drop_index(offsets.astype(np.uint32)),
            "count": drop_index(lengths.astype(np.uint32)),
        }), on="bid", how="left")

        # Determine each edge's index relative to its bundle
        edges = drop_index(edges.sort_values(by="bid"))
        edges["index"] = edges.index.to_series() - edges["start"]
        edges["index"] = edges["index"].astype(np.uint32)

        # Re-sort the edgelist by edge id and cleanup
        edges = drop_index(edges.sort_values(by="eid"))
        edges = edges.rename({"eid": "id"}, axis=1, copy=False)
        edges = edges[["id", "src", "dst", "index", "count"]]

        return {
            "edge": smoosh(edges[["src", "dst"]]).astype(np.uint64),
            "bundle": smoosh(edges[["index", "count"]]).astype(np.uint64),
        }

    def make_nodes(df, src, dst):
        nodes = drop_index(df[src].append(df[dst], ignore_index=True).unique())
        ids = drop_index(cudf.Series(nodes.factorize()[0])).astype(np.uint32)
        return drop_index(cudf.DataFrame({"id": ids, "node": nodes}).sort_values(by="id"))

    def make_edges(df, src, dst, nodes):
        def join(edges, nodes, col):
            edges = edges.set_index(col, drop=True)
            nodes = nodes.set_index("node", drop=True)
            edges = edges.join(nodes).sort_values(by="eid")
            edges = edges.rename({"id": col}, axis=1, copy=False)
            return drop_index(edges)
        edges = df.reset_index().rename({"index": "eid"}, axis=1, copy=False)
        edges = join(join(edges.assign(src=df[src], dst=df[dst]), nodes, "src"), nodes, "dst")
        edges = edges.assign(**compute_edge_bundles(edges, "eid", "src", "dst"))
        return drop_index(edges.rename({"eid": "id"}, axis=1, copy=False))

    df = drop_index(df)
    nodes = make_nodes(df, source, target)
    edges = make_edges(df, source, target, nodes)
    graph = cugraph.MultiGraph(directed=True)
    graph._Impl = simpleGraphImpl(graph.graph_properties)
    graph._Impl.edgelist = EdgeList(edges["src"], edges["dst"])
    degree = graph._Impl.degree().set_index("vertex")
    nodes = nodes.set_index("id", drop=False).join(degree)
    return graph, drop_index(nodes.sort_index()), edges
