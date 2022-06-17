'use strict'

const {dir}                                   = require('console');
const {test}                                  = require('tap');
const {build}                                 = require('../helper');
const {tableFromIPC, RecordBatchStreamWriter} = require('apache-arrow');

const json_large = {
  'json_large.txt':
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
            "size": 0,
            "label": "Node n°292, in cluster n°1",
            "color": "#323455"
          }
        }
      ],
      "edges": [
        {"key": "geid_115_98", "source": "128", "target": "193"},
        {"key": "geid_115_99", "source": "269", "target": "93"}
      ],
      "options": {"type": "mixed", "multi": false, "allowSelfLoops": true}
    }`
};

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

test('graphology root returns api description', async t => {
  const app = await build(t);
  const res = await app.inject({url: '/graphology'})
  t.same(JSON.parse(res.payload), {
    graphology: {
      description: 'The graphology api provides GPU acceleration of graphology datasets.',
      schema: {
        read_json: {
          filename: 'A URI to a graphology json dataset file.',
          result: `Causes the node-rapids backend to attempt to load the json object specified
                     by :filename. The GPU with attempt to parse the json file asynchronously and will
                     return OK/ Not Found/ or Fail based on the file status.
                     If the load is successful, three tables will be created in the node-rapids backend:
                     nodes, edges, clusters, and tags. The root objects in the json target must match
                     these names and order.`,
          returns: 'Result OK/Not Found/Fail'
        },
        read_large_demo: {
          filename:
            'A URI to a graphology json dataset file matching the sigma.js/examples/large-demos spec.',
          result: `Produces the same result as 'read_json'.
                     If the load is successful, three tables will be created in the node-rapids backend:
                     nodes, edges, and options.`,
          returns: 'Result OK/Not Found/Fail'
        },
        list_tables: {returns: 'Tables that are available presently in GPU memory.'},
        get_table: {
          ':table':
            {table: 'The name of the table that has been allocated previously into GPU memory.'}
        },
        get_column: {':table': {':column': {table: 'The table name', column: 'The column name'}}},
        nodes: {
          returns:
            'Returns the existing nodes table after applying normalization functions for sigma.js'
        },
        nodes: {bounds: {returns: 'Returns the x and y bounds to be used in rendering.'}},
        edges:
          {return: 'Returns the existing edges table after applying normalization for sigma.js'}
      }
    }
  })
});

test('read_json no filename', async t => {
  const app = await build(t);
  const res = await app.inject({method: 'POST', url: '/graphology/read_json'});
  t.same(
    JSON.parse(res.payload),
    {success: false, message: 'Parameter filename is required', params: '{}', statusCode: 400});
});

test('read_json no file', async (t) => {
  const app = await build(t);
  const res =
    await app.inject({method: 'POST', url: '/graphology/read_json?filename=filenotfound.txt'});
  const payload = JSON.parse(res.payload);
  t.ok(payload.message.includes('no such file or directory'));
  t.equal(payload.statusCode, 404);
});

test('read_json incorrect format', async (t) => {
  const dir   = t.testdir({
    'json_bad.txt': ` {
            [
              {
                "key": "customer data management",
                "label": "Customer data management",
                "cluster": "7",
                "x": -278.2200012207031,
                "y": 436.0100402832031,
                "score": 0
              },
              {
                "key": "educational data mining",
                "label": "Educational data mining",
                "cluster": "7",
                "x": -1.9823756217956543,
                "y": 250.4990692138672,
                "score": 0
              },
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
  });
  const rpath = '../../test/routes/' + dir.substring(dir.lastIndexOf('/')) + '/json_bad.txt';
  const app   = await build(t);
  const res   = await app.inject({method: 'POST', url: '/graphology/read_json?filename=' + rpath});
  const release = await app.inject({method: 'POST', url: '/graphology/release'});
  const payload = JSON.parse(res.payload);
  t.equal(payload.message, 'Bad graphology format: nodes not found.');
  t.equal(payload.success, false);
});

test('read_json file good', async (t) => {
  const dir   = t.testdir(json_good);
  const rpath = '../../test/routes/' + dir.substring(dir.lastIndexOf('/')) + '/json_good.txt';
  const app   = await build(t);
  const res   = await app.inject({method: 'POST', url: '/graphology/read_json?filename=' + rpath});
  const release = await app.inject({method: 'POST', url: '/graphology/release'});
  const payload = JSON.parse(res.payload);
  t.equal(payload.message, 'File read onto GPU.');
  t.equal(payload.success, true);
});

test('list_tables', async (t) => {
  const dir   = t.testdir(json_good);
  const rpath = '../../test/routes/' + dir.substring(dir.lastIndexOf('/')) + '/json_good.txt';
  const app   = await build(t);
  const load  = await app.inject({method: 'POST', url: '/graphology/read_json?filename=' + rpath});
  const res   = await app.inject({method: 'GET', url: '/graphology/list_tables'});
  const release = await app.inject({method: 'POST', url: '/graphology/release'});
  const payload = JSON.parse(res.payload);
  t.ok(payload.includes('nodes'));
});

test('get_table', async (t) => {
  const dir   = t.testdir(json_good);
  const rpath = '../../test/routes/' + dir.substring(dir.lastIndexOf('/')) + '/json_good.txt';
  const app   = await build(t);
  const load  = await app.inject({method: 'POST', url: '/graphology/read_json?filename=' + rpath});
  const res   = await app.inject({
    method: 'GET',
    url: '/graphology/get_table/nodes',
    header: {'accepts': 'application/octet-stream'}
  });
  const table = tableFromIPC(res.rawPayload);
  const release = await app.inject({method: 'POST', url: '/graphology/release'});
  t.same(table.schema.names, ['key', 'label', 'tag', 'URL', 'cluster', 'x', 'y', 'score']);
  t.equal(table.numRows, 2);
  t.equal(table.numCols, 8);
});

test('get_column', async (t) => {
  const dir   = t.testdir(json_good);
  const rpath = '../../test/routes/' + dir.substring(dir.lastIndexOf('/')) + '/json_good.txt';
  const app   = await build(t);
  const load  = await app.inject({method: 'POST', url: '/graphology/read_json?filename=' + rpath});
  const res   = await app.inject({
    method: 'GET',
    url: '/graphology/get_column/nodes/score',
    header: {'accepts': 'application/octet-stream'}
  });
  const table = tableFromIPC(res.rawPayload);
  const release = await app.inject({method: 'POST', url: '/graphology/release'});
  t.same(table.schema.names, ['score']);
  t.equal(table.numRows, 2);
  t.equal(table.numCols, 1);
});

test('nodes', async (t) => {
  const dir   = t.testdir(json_large);
  const rpath = '../../test/routes/' + dir.substring(dir.lastIndexOf('/')) + '/json_large.txt';
  const app   = await build(t);
  const load =
    await app.inject({method: 'POST', url: '/graphology/read_large_demo?filename=' + rpath});
  const res = await app.inject(
    {method: 'GET', url: '/graphology/nodes/', headers: {'accepts': 'application/octet-stream'}});
  const table = tableFromIPC(res.rawPayload);
  t.ok(table.getChild('nodes'));
});

test('nodes/bounds', async (t) => {
  const dir   = t.testdir(json_good);
  const rpath = '../../test/routes/' + dir.substring(dir.lastIndexOf('/')) + '/json_good.txt';
  const app   = await build(t);
  const load  = await app.inject({method: 'POST', url: '/graphology/read_json?filename=' + rpath});
  const res   = await app.inject({method: 'GET', url: '/graphology/nodes/bounds'});
  const release = await app.inject({method: 'POST', url: '/graphology/release'});
  t.same(JSON.parse(res.payload), {
    'success': true,
    'message': 'Success',
    'bounds': {
      'xmin': -278.2200012207031,
      'xmax': -1.9823756217956543,
      'ymin': 250.4990692138672,
      'ymax': 436.01004028320307
    }
  });
});

test('edges', async (t) => {
  const dir   = t.testdir(json_large);
  const rpath = '../../test/routes/' + dir.substring(dir.lastIndexOf('/')) + '/json_large.txt';
  const app   = await build(t);
  const load =
    await app.inject({method: 'POST', url: '/graphology/read_large_demo?filename=' + rpath});
  const res = await app.inject(
    {method: 'GET', url: '/graphology/edges', header: {'accepts': 'application/octet-stream'}});
  const table   = tableFromIPC(res.rawPayload);
  const release = await app.inject({method: 'POST', url: '/graphology/release'});
  t.ok(table.getChild('edges'));
});
