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
from .convert_matrix import from_cudf_edgelist
from .graph_components import (
    annotate_nodes,
    annotate_edges,
    category_to_color
)


def make_synthetic_dataset(**kwargs):
    import pandas as pd
    import datetime as dt
    kwargs.update(direct=True)
    df = cudf.DataFrame.from_pandas(pd.DataFrame({
        "src": [0, 1, 2, 3],
        "dst": [1, 2, 3, 0],
        "colors": [1, 1, 2, 2],
        "bool": [True, False, True, True],
        "char": ["a", "b", "c", "d"],
        "str": ["a", "b", "c", "d"],
        "ustr": [u"a", u"b", u"c", u"d"],
        "emoji": ["😋", "😋😋", "😋", "😋"],
        "int": [0, 1, 2, 3],
        "num": [0.5, 1.5, 2.5, 3.5],
        "date_str": [
            "2018-01-01 00:00:00",
            "2018-01-02 00:00:00",
            "2018-01-03 00:00:00",
            "2018-01-05 00:00:00",
        ],
        "date": [
            dt.datetime(2018, 1, 1),
            dt.datetime(2018, 1, 1),
            dt.datetime(2018, 1, 1),
            dt.datetime(2018, 1, 1),
        ],
        "time": [
            pd.Timestamp("2018-01-05"),
            pd.Timestamp("2018-01-05"),
            pd.Timestamp("2018-01-05"),
            pd.Timestamp("2018-01-05"),
        ],
    }))
    return make_and_shape_hypergraph(df, **kwargs)


def make_and_shape_hypergraph(df, **kwargs):
    hyper = cugraph.hypergraph(df, **kwargs)
    del hyper["events"]
    del hyper["entities"]
    SOURCE = kwargs.get("SOURCE", "src")
    TARGET = kwargs.get("TARGET", "dst")
    NODEID = kwargs.get("NODEID", "node_id")
    EVENTID = kwargs.get("EVENTID", "event_id")
    CATEGORY = kwargs.get("CATEGORY", "category")
    nodes = hyper["nodes"][[NODEID, CATEGORY]]
    edges = hyper["edges"][[SOURCE, TARGET]]
    # Create graph
    graph, nodes, edges = from_cudf_edgelist(edges, SOURCE, TARGET)
    nodes["name"] = nodes["node"]
    # Add vis components 
    nodes = annotate_nodes(graph, nodes, edges)
    edges = annotate_edges(graph, nodes, edges)
    return graph, nodes, edges


def make_capwin_dataset(**kwargs):

    def drop_index(df):
        return df.reset_index(drop=True)

    def smoosh(df):
        size = sum([df[x].dtype.itemsize for x in df])
        data = drop_index(drop_index(df).stack()).data
        dtype = cudf.utils.dtypes.min_unsigned_type(0, size*8)
        return cudf.core.column.NumericalColumn(data, dtype=dtype)

    def add_edge_colors(edges, category):
        colors = drop_index(category_to_color(edges[category], color_palette=[
            #  ADDRESS   AUTH KEYS CREDENTIALS       EMAIL      FALSE
            4294967091, 4294410687, 4293138972, 4281827000,  33554431
        ]).astype(np.uint32))
        return edges.assign(color=smoosh(cudf.DataFrame({
            "src": drop_index(colors), "dst": drop_index(colors)
        })).astype(np.uint64), src_color=colors)
    
    df = cudf.read_csv("data/pii_sample_for_viz.csv")
    df = df[["src_ip", "dest_ip", "pii", "timestamp"]]
    df["timestamp"] = cudf.to_datetime(df["timestamp"], format="%m/%d/%y %H:%M")
    # Create graph
    graph, nodes, edges = from_cudf_edgelist(df, "src_ip", "dest_ip")
    # Add vis components 
    nodes = nodes.rename({"node": "name"}, axis=1, copy=False)
    nodes = annotate_nodes(graph, nodes, edges)
    # add edge colors
    edges = add_edge_colors(edges, "pii")
    print(edges.query("src_color != 33554431")["src"].value_counts())
    print(edges.query("src_color != 33554431")["dst"].value_counts())
    # add edge names
    edges["name"] = edges["src_ip"] + " -> " + edges["dest_ip"] + \
        ("\nPII: " + edges["pii"]).replace("\nPII: FALSE", "")
    return graph, nodes, edges
