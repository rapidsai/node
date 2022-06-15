'use strict'

const {test}         = require('tap')
const Fastify        = require('fastify')
const Support        = require('../../plugins/support')
const {StringSeries} = require('@rapidsai/cudf');

test('clearCachedGPUData()', async (t) => {
  const gpuCache = require('../../util/gpu_cache.js');
  gpuCache.setDataframe('bob', 5);
  t.equal(gpuCache.getDataframe('bob'), 5);
});

test('readLargeGraphDemo', (t) => {
  const gpuCache =
    t.mock('../../util/gpu_cache.js', {'StringSeries': {read_text: (path, delim) => path}})
  gpuCache.readLargeGraphDemo(
    `{
    'attributes': {},
      'nodes':
        [
          {
            'key': '0',
            'attributes': {
              'cluster': 1,
              'x': 2.1690608678749332,
              'y': -0.3294237082577565,
              'size': 0.3333333333333333,
              'label': 'Node n°1, in cluster n°1',
              'color': '#486add'
            }
          },
          {
            'key': '1',
            'attributes': {
              'cluster': 2,
              'x': 1.535086271659372,
              'y': 1.5674358572750524,
              'size': 1,
              'label': 'Node n°2, in cluster n°2',
              'color': '#71b952'
            }
          },
        ],
      'edges':
        [
          {'key': 'geid_99_0', 'source': '2', 'target': '1'},
          {'key': 'geid_99_1', 'source': '4', 'target': '1'},
        ],
      'options': {'type': 'mixed', 'multi': false, 'allowSelfLoops': true}
}`)
});
  test('readGraphology', async (t) => {
    const gpuCache =
      t.mock('../../util/gpu_cache.js', {StringSeries: {read_text: (path, delim) => path}})
    gpuCache.readGraphology(
      `
{
  "nodes": [
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
    },
  ],
  "edges": [
    ["office suite", "human interactome"],
    ["educational data mining", "human interactome"],
  ],
  "clusters": [
    { "key": "0", "color": "#6c3e81", "clusterLabel": "human interactome" },
    { "key": "1", "color": "#666666", "clusterLabel": "Spreadsheets" },
  ],
  "tags": [
    { "key": "Chart type", "image": "charttype.svg" },
    { "key": "Company", "image": "company.svg" },
  ]
}`);
  });
