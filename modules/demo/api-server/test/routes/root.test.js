'use strict'

const {test}  = require('tap');
const {build} = require('../helper');

test('root returns API description', async (t) => {
  const app = await build(t);
  const res = await app.inject({url: '/'});
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
  });
});
