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

const json_good = {
  'json_good.txt':
    ` {
          "nodes":
            [
              {
                "key": "customer data management",
                "label": "Customer data management",
                "tag": "Field",
                "URL": "https://en.wikipedia.org/wiki/Customer%20data%20management",
                "cluster": "7",
                "x": -278.2200012207031,
                "y": 436.0100402832031,
                "score": 0
              },
              {
                "key": "educational data mining",
                "label": "Educational data mining",
                "tag": "Field",
                "URL": "https://en.wikipedia.org/wiki/Educational%20data%20mining",
                "cluster": "7",
                "x": -1.9823756217956543,
                "y": 250.4990692138672,
                "score": 0
              }
            ],
            "edges":
              [
                ["office suite", "human interactome"],
                ["educational data mining", "human interactome"],
              ],
            "clusters":
              [
                {"key": "0", "color": "#6c3e81", "clusterLabel": "human interactome"},
                {"key": "1", "color": "#666666", "clusterLabel": "Spreadsheets"},
              ],
            "tags": [
              {"key": "Chart type", "image": "charttype.svg"},
              {"key": "Company", "image": "company.svg"},
            ]
        } `
};

const json_large = {
  'json_large.txt':
    ` {
      "attributes": {},
      "nodes": [
        {
          "key": "0",
          "attributes": {
            "cluster": 0,
            "x": -13.364310772761677,
            "y": 4.134339113107921,
            "size": 0,
            "label": "Node n°291, in cluster n°0",
            "color": "#e24b04"
          }
        },
        {
          "key": "1",
          "attributes": {
            "cluster": 1,
            "x": 1.3836898237261988,
            "y": -11.536596764896206,
            "size": 1,
            "label": "Node n°292, in cluster n°1",
            "color": "#323455"
          }
        }
      ],
      "edges": [
        {"key": "geid_115_98", "source": "1", "target": "0"},
        {"key": "geid_115_99", "source": "0", "target": "1"}
      ],
      "options": {"type": "mixed", "multi": false, "allowSelfLoops": true}
    }`
};

const json_out_of_order = {
  'json_out_of_order.txt':
    ` {
      "attributes": {},
      "nodes": [
        {
          "key": "290",
          "attributes": {
            "cluster": 0,
            "x": -13.364310772761677,
            "y": 4.134339113107921,
            "size": 0,
            "label": "Node n°291, in cluster n°0",
            "color": "#e24b04"
          }
        },
        {
          "key": "291",
          "attributes": {
            "cluster": 1,
            "x": 1.3836898237261988,
            "y": -11.536596764896206,
            "size": 1,
            "label": "Node n°292, in cluster n°1",
            "color": "#323455"
          }
        }
      ],
      "edges": [
        {"key": "geid_115_98", "source": "0", "target": "1"},
        {"key": "geid_115_99", "source": "1", "target": "0"}
      ],
      "options": {"type": "mixed", "multi": false, "allowSelfLoops": true}
    }`
};

module.exports = {
  json_good: json_good,
  json_large: json_large,
  json_out_of_order: json_out_of_order
};
