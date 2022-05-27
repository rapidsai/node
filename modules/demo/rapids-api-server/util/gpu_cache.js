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

const {Bool8, Utf8String, Int32, Int64, DataFrame, StringSeries, Series, Float32, Float64} =
  require('@rapidsai/cudf');

let timeout  = -1;
let datasets = {};

function clearCachedGPUData() {
  for (const key in datasets) { datasets[key] = null; }
};

function json_key_attributes_to_dataframe(str) {
  let arr       = {};
  const columns = ['cluster', 'x', 'y', 'size', 'label', 'color'];
  const dtypes  = [new Int32, new Float32, new Float32, new Int32, new Utf8String, new Utf8String];
  const no_open_list = str.split('[\n').gather([1], false);
  const tokenized    = no_open_list.split('},');
  columns.forEach((col, ix) => {
    const parse_result = tokenized._col.getJSONObject('.attributes.' + columns[ix]);
    const string_array = Series.new(parse_result);
    arr[col]           = string_array.cast(dtypes[ix]);
  });
  const result = new DataFrame(arr);
  return result;
}

function json_aos_to_dataframe(str, columns, dtypes) {
  let arr = {};
  columns.forEach((col, ix) => {
    const no_open_list = str.split('[\n').gather([1], false);
    const tokenized    = no_open_list.split('},');
    const parse_result = tokenized._col.getJSONObject('.' + columns[ix]);
    arr[col]           = Series.new(parse_result).cast(dtypes[ix]);
  });
  const result = new DataFrame(arr);
  return result;
}

function json_aoa_to_dataframe(str, dtypes) {
  let arr            = {};
  const no_open_list = str.split('[\n').gather([1], false);
  const tokenized    = no_open_list.split('],');
  dtypes.forEach((_, ix) => {
    const get_ix       = `[${ix}]`;
    const parse_result = tokenized._col.getJSONObject(get_ix);
    const string_array = Series.new(parse_result);
    arr[ix]            = string_array.cast(dtypes[ix]);
  });
  const result = new DataFrame(arr);
  return result;
}

function getGraphologyObjectOrder(json_dataset) {
  const graphologyObjects =
    ['"nodes:"', '"clusters:"', '"tags:"', '"edges:"', '"attributes:"', '"options:"'];
  graphologyObjects.forEach((key, ix) => {
    console.log(key);
    console.log(ix);
    console.log({key: key, ix: ix, pos: json_dataset._col.find(key)});
  });
}

module.exports = {
  setDataframe(name, dataframe) {
    if (timeout) { clearTimeout(timeout); }
    timeout        = setTimeout(clearCachedGPUData, 10 * 60 * 1000);
    datasets[name] = dataframe;
    console.log(datasets);
  },
  getDataframe(name) { return datasets[name] },
  readLargeGraphDemo(path) {
    console.log('readLargeGraphDemo');
    const dataset  = StringSeries.readText(path, '');
    let split      = dataset.split('"options":');
    const toptions = split.gather([1], false);
    let rest       = split.gather([0], false);
    split          = rest.split('"edges":');
    const tedges   = split.gather([1], false);
    rest           = split.gather([0], false);
    split          = rest.split('"nodes":');
    const tnodes   = split.gather([1], false);
    const nodes    = json_key_attributes_to_dataframe(tnodes);
    const edges    = json_aos_to_dataframe(
      tedges, ['key', 'source', 'target'], [new Utf8String, new Int32, new Int32]);
    let optionsArr               = {};
    optionsArr['type']           = Series.new(toptions._col.getJSONObject('.type'));
    optionsArr['multi']          = Series.new(toptions._col.getJSONObject('.multi'));
    optionsArr['allowSelfLoops'] = Series.new(toptions._col.getJSONObject('.allowSelfLoops'));
    const options                = new DataFrame(optionsArr);
    return {nodes: nodes, edges: edges, options: options};
  },
  readGraphology(path) {
    console.log('readGraphology');
    const dataset = StringSeries.readText(path, '');
    getGraphologyObjectOrder(dataset);
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
