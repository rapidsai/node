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
import pandas as pd
import datetime as dt
from .convert_matrix import from_cudf_edgelist
from .graph_components import (
    annotate_nodes,
    annotate_edges
)


def make_synthetic_dataset(**kwargs):
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
