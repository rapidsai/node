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

const fastify   = require('fastify');

module.exports = async function(fastify, opts) {
  fastify.get('/', async function(request, reply) {
    return {
      graphology: {
        description: 'The graphology api provides GPU acceleration of graphology datasets.',
        schema: {
          read_json: {
            filename: 'A URI to a graphology json dataset file.',
            returns: 'Result OK/Not Found/Fail'
          },
          list_tables:
            {returns: 'An object containing graphology related datasets resident on GPU memory.'},

          ':table': {
            ':column': 'The name of the column you want to request.',
            returns: 'An arrow buffer of the column contents.'
          }
        }
      }
    }
  });
}
