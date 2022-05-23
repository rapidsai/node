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

const {Utf8String, Int32, DataFrame, StringSeries, Series, Float64} = require('@rapidsai/cudf');

let timeout  = -1;
let datasets = {};

function clearCachedGPUData() {
  for (const key in datasets) { datasets[key] = null; }
};

/* TODO: How do I apply a list of dtypes?
 */
function json_aos_to_dataframe(str, columns, _) {
  let arr = {};
  columns.forEach((col, ix) => {
    const no_open_list = str.split('[\n').gather([1], false);
    const tokenized    = no_open_list.split('},');
    const parse_result = tokenized._col.getJSONObject('.' + columns[ix]);
    arr[col]           = Series.new(parse_result);
  });
  const result = new DataFrame(arr);
  return result;
}
/* TODO: How do I apply a list of dtypes?
 */
function json_aoa_to_dataframe(str, dtypes) {
  let arr = {};
  dtypes.forEach((_, ix) => {
    const no_open_list = str.split('[\n').gather([1], false);
    const tokenized    = no_open_list.split('],');
    const get_ix       = `[${ix}]`;
    const parse_result = tokenized._col.getJSONObject(get_ix);
    arr[ix]            = Series.new(parse_result);
  });
  const result = new DataFrame(arr);
  return result;
}

module.exports = {
  setDataframe(name, dataframe) {
    if (timeout) { clearTimeout(timeout); }
    timeout        = setTimeout(clearCachedGPUData, 10 * 60 * 1000);
    datasets[name] = dataframe;
    console.log(datasets);
  },
  getDataframe(name) { return datasets[name] },
  readGraphology(path) {
    console.log('readGraphology');
    const dataset   = StringSeries.readText(path, '');
    let split       = dataset.split('"tags":');
    const ttags     = split.gather([1], false);
    let rest        = split.gather([0], false);
    split           = rest.split('"clusters":');
    const tclusters = split.gather([1], false);
    rest            = split.gather([0], false);
    split           = rest.split('"edges":');
    const tedges    = split.gather([1], false);
    rest            = split.gather([0], false);
    split           = rest.split('"nodes":');
    const tnodes    = split.gather([1], false);
    const tags = json_aos_to_dataframe(ttags, ['key', 'image'], [new Utf8String, new Utf8String]);
    const clusters = json_aos_to_dataframe(
      tclusters, ['key', 'color', 'clusterLabel'], [new Int32, new Utf8String, new Utf8String]);
    const nodes =
      json_aos_to_dataframe(tnodes, ['key', 'label', 'tag', 'URL', 'cluster', 'x', 'y', 'score'], [
        new Utf8String,
        new Utf8String,
        new Utf8String,
        new Utf8String,
        new Int32,
        new Float64,
        new Float64,
        new Int32
      ]);
    const edges = json_aoa_to_dataframe(tedges, [new Utf8String, new Utf8String]);
    return {nodes: nodes, edges: edges, tags: tags, clusters: clusters};
  }
}
