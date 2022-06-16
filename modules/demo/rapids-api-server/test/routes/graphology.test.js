'use strict'

const {test}  = require('tap')
const {build} = require('../helper')

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
  t.same(JSON.parse(res.payload),
         {success: false, message: 'Parameter filename is required', params: '{}'});
});

test('read_json no file', async (t) => {
  const app = await build(t);
  const res =
    await app.inject({method: 'POST', url: '/graphology/read_json?filename=filenotfound.txt'});
  const payload = JSON.parse(res.payload);
  t.ok(payload.message.includes('no such file or directory'));
  t.equal(payload.statusCode, 500);
});

test('read_json incorrect format', {only: true, saveFixture: true}, async (t) => {
  const dir   = t.testdir({
    'json_bad.txt': ` {
          "nodes":
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
  const release = await app.inject({method: 'POST', url: '/release'});
  const payload = JSON.parse(res.payload);
  t.equal(payload.message, 'File read onto GPU.');
  t.equal(payload.success, true);
});

test('read_json file good', async (t) => {
  const dir   = t.testdir({
    'json_good.txt': ` {
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
  const rpath = '../../test/routes/' + dir.substring(dir.lastIndexOf('/')) + '/json_good.txt';
  const app   = await build(t);
  const res   = await app.inject({method: 'POST', url: '/graphology/read_json?filename=' + rpath});
  const release = await app.inject({method: 'POST', url: '/release'});
  const payload = JSON.parse(res.payload);
  t.equal(payload.message, 'File read onto GPU.');
  t.equal(payload.success, true);
});
