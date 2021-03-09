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
from math import ceil


def annotate_nodes(graph, nodes, edges):
    return nodes.assign(
        # add node names
        name=nodes["name"] if "name" in nodes else nodes["id"],
        # add node sizes
        size=(nodes["degree"].scale() * 254 + 1).astype(np.uint8),
        # add node colors
        color=category_to_color(
            cugraph.spectralBalancedCutClustering(graph, 9)
                   .sort_values(by="vertex").reset_index(drop=True)["cluster"],
            color_palette=[
                # https://colorbrewer2.org/#type=diverging&scheme=Spectral&n=9
                4292165199, 4294208835,
                4294815329, 4294893707,
                4294967231, 4293326232,
                4289453476, 4284924581,
                4281501885
            ])
    )


def annotate_edges(graph, nodes, edges):
    def drop_index(df):
        return df.reset_index(drop=True)

    def smoosh(df):
        size = sum([df[x].dtype.itemsize for x in df])
        data = drop_index(drop_index(df).stack()).data
        dtype = cudf.utils.dtypes.min_unsigned_type(0, size*8)
        return cudf.core.column.NumericalColumn(data, dtype=dtype)

    def edge_colors(nodes, edges, col):
        edges = edges[["id", col]].set_index(col, drop=True)
        nodes = nodes[["id", "color"]].set_index("id", drop=True)
        return drop_index(edges.join(nodes).sort_values(by="id")["color"])

    return edges.assign(
        # add edge names
        name=edges["name"] if "name" in edges else edges["id"],
        # add edge colors
        color=smoosh(cudf.DataFrame({
            "src": edge_colors(nodes, edges, "src"),
            "dst": edge_colors(nodes, edges, "dst"),
        })))


# from random import shuffle
default_palette = [
    # https://colorbrewer2.org/#type=diverging&scheme=Spectral&n=11
    4288545090, 4292165199,
    4294208835, 4294815329,
    4294893707, 4294967231,
    4293326232, 4289453476,
    4284924581, 4281501885,
    4284370850
    # -12451426,-11583787,-12358156,-10375427,
    # -7610114,-4194305,-6752794,-5972565,
    # -5914010,-4356046,-6140066
]
# shuffle(color_palette)

def category_to_color(categories, color_palette=None):
    if color_palette is None:
        color_palette = default_palette
    color_indices = cudf.Series(categories)
    color_palette = cudf.Series(color_palette)
    if color_indices.dtype.type != np.uint32:
        color_indices = cudf.Series(categories.factorize()[0]).astype(np.uint32)
    color_palettes = []
    num_color_ids = color_indices.max() + 1
    for i in range(ceil(num_color_ids / len(color_palette))):
        color_palettes.append(color_palette)
    return cudf.Series(cudf.core.column.build_categorical_column(
        ordered=True,
        codes=color_indices._column,
        categories=cudf.concat(color_palettes)[:num_color_ids],
    ).as_numerical_column(dtype=np.uint32))
