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
import pandas as pd
import datetime as dt

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)


def make_small_dataframe(**kwargs):
    df = cudf.DataFrame({
        "col_1": ["a", "a", "a"],
        "col_2": ["a", "b", "b"],
        "col_3": ["a", "b", "c"],
    })
    return df, kwargs


def make_large_dataframe(**kwargs):
    df = cudf.read_csv(
        'data/Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv',
        # '/home/ptaylor/dev/rapids/compose/etc/notebooks/sandbox/data/Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv',
        parse_dates=[2],
    ).dropna()
    df.reset_index(drop=True, inplace=True)
    df['Label'] = df['Label'].astype('category')
    df = df[['Label', 'Timestamp', 'Dst Port', 'Protocol']]
    kwargs.update(EVENTID='Dst Port')
    kwargs.update(SKIP=['Timestamp'])
    kwargs.update(drop_edge_attrs=True)
    return df, kwargs


def make_complex_dataframe(**kwargs):
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
    return df, kwargs
