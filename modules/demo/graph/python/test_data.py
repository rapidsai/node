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


import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import cudf
import cupy
import cugraph
import numpy as np
import pandas as pd
import datetime as dt
from .shaping import shape_graph, _filter_by


def make_small_dataset(**kwargs):
    df = cudf.DataFrame({
        "col_1": ["a", "a", "a"],
        "col_2": ["a", "b", "b"],
        "col_3": ["a", "b", "c"],
    })
    return make_and_shape_hypergraph(df, **kwargs)


def make_large_dataset(**kwargs):
    df = cudf.read_csv(
        "data/Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv",
        parse_dates=[2],
    ).dropna()
    df.reset_index(drop=True, inplace=True)
    df["Label"] = df["Label"].astype("category")
    df = df[["Label", "Timestamp", "Dst Port", "Protocol"]]
    kwargs.update(EVENTID="Dst Port")
    kwargs.update(SKIP=["Timestamp"])
    kwargs.update(drop_edge_attrs=True)
    return make_and_shape_hypergraph(df, **kwargs)


def make_complex_dataset(**kwargs):
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


def make_cit_patents_dataset(**kwargs):
    graph = cugraph.from_cudf_edgelist(
        cudf.read_csv(
            "data/cit-Patents.csv",
            delimiter=" ",
            dtype=["int32", "int32", "float32"],
            header=None
        ),
        source="0", destination="1",
        create_using=cugraph.structure.graph.DiGraph,
        renumber=True
    )
    nodes = graph.nodes()
    categories = cupy.repeat(cupy.arange(12), 1 + (len(nodes) // 12))
    kwargs.update(graph=graph)
    kwargs.update(nodes=cudf.DataFrame({
        "node_id": nodes,
        "category": categories[:len(nodes)]
    }))
    return shape_graph(**kwargs, symmetrize=False)


def make_capwin_dataset(**kwargs):
    graph = cudf.read_csv("data/capWIN-friday-biflows.csv", parse_dates=[7])
    # [print(str(x) + ': ' + str(y)) for x, y in zip(graph.columns, graph.dtypes)]
    graph = cugraph.from_cudf_edgelist(
        graph, source="Src IP", destination="Dst IP",
        create_using=cugraph.structure.graph.DiGraph,
        renumber=True
    )
    nodes = graph.nodes()
    labels = cugraph.core_number(graph)
    labels = _filter_by(labels, nodes, left_on="vertex")
    labels = labels["core_number"]
    kwargs.update(graph=graph)
    kwargs.update(nodes=cudf.DataFrame({
        "node_id": nodes, "category": labels,
    }))
    return shape_graph(**kwargs, symmetrize=False)
    # kwargs.update(SKIP=["Timestamp", "Src Port", "Dst Port"])
    # kwargs.update(drop_edge_attrs=True)
    # kwargs.update(categories={
    #     "Src Port": "Port",
    #     "Dst Port": "Port",
    # })
    # return make_and_shape_hypergraph(df, **kwargs)


def make_and_shape_hypergraph(df, **kwargs):
    xs = cugraph.hypergraph(df, **kwargs)
    del xs["events"]
    del xs["entities"]
    NODEID = kwargs.get("NODEID", "node_id")
    CATEGORY = kwargs.get("CATEGORY", "category")
    xs.update(nodes=xs["nodes"][[NODEID, CATEGORY]])
    return shape_graph(symmetrize=False, **xs, **kwargs)
