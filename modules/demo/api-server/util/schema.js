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

'use strict';

const schema = {
  gpu: {
    description: 'An abstract interface to the node-rapids api, supported by a server.',
    schema: {
      '/': {
        method: 'The name of the method to apply to gpu_cache data.',
        caller: 'Either an object that has been stored in the gpu_cache or a static module name.',
        arguments: 'Correctly specified arguments to the gpu_cache method.',
        result: 'Either a result code specifying success or failure or an arrow data buffer.',
      }
    }
  },
  graphology: {
    description: 'The graphology api provides GPU acceleration of graphology datasets.',
    schema: {
      read_json: {
        filename: 'A URI to a graphology json dataset file.',
        result: `Causes the node-rapids backend to attempt to load the json object specified
                     by :filename. The GPU will attempt to parse the json file asynchronously and will
                     return OK/ Not Found/ or Fail based on the file status.
                     If the load is successful, four tables will be created in the node-rapids backend:
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
      edges: {return: 'Returns the existing edges table after applying normalization for sigma.js'}
    }
  }
};

module.exports = schema;
