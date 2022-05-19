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

const Fs                                                            = require('fs');
const {Utf8String, Int32, DataFrame, StringSeries, Series, Float64} = require('@rapidsai/cudf');
const {RecordBatchStreamWriter, Field, Vector, List, Table}         = require('apache-arrow');
const Path                                                          = require('path');
const {promisify}                                                   = require('util');
const Stat                                                          = promisify(Fs.stat);

const fastify     = require('fastify');
const arrowPlugin = require('fastify-arrow');
const gpu_cache   = require('../../util/gpu_cache.js');

module.exports = async function(fastify, opts) {
  fastify.register(arrowPlugin);
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

  fastify.route({
    method: 'GET',
    url: '/read_json',
    schema: {
      querystring: {filename: {type: 'string'}, 'rootkeys': {type: 'array'}},

      response: {
        200: {
          type: 'object',
          properties:
            {success: {type: 'boolean'}, message: {type: 'string'}, params: {type: 'string'}}
        }
      }
    },
    handler: async (request, reply) => {
      // load the file via read_text
      // is the file local or remote?
      let message = 'Unknown error';
      let result  = {'params': JSON.stringify(request.query), success: false, message: message};

      console.log(result);
      if (request.query.filename.search('http') != -1) {
        message = 'Remote files not supported yet.'
        console.log(result);
      } else {
        message = 'File is not remote';
        console.log(message);
        // does the file exist?
        const path = Path.join(__dirname, request.query.filename);
        console.log('Does the file exist?');
        try {
          const stats = await Stat(path);
          if (stats == null) {
            message = 'File does not exist at ' + path;
            console.log(message);
            result.message = message;
            reply.code(200).send(result);
          } else {
            // did the file read?
            message = 'File is available';
            console.log(message);
            result.success   = true;
            message          = 'Successfully parsed json file onto GPU.';
            result.message   = message;
            const graphology = gpu_cache.readGraphology(path);
            gpu_cache.setDataframe('nodes', graphology['nodes']);
            gpu_cache.setDataframe('edges', graphology['edges']);
            gpu_cache.setDataframe('clusters', graphology['clusters']);
            gpu_cache.setDataframe('tags', graphology['tags']);
            reply.code(200).send(result);
          }
        } catch {
          message        = 'Exception reading file.';
          result.message = message;
          reply.code(200).send(result);
        };
      }
    }
  });

  fastify.route({
    method: 'GET',
    url: '/get_column/:table/:column',
    schema: {querystring: {table: {type: 'string'}, 'column': {type: 'string'}}},
    handler: async (request, reply) => {
      let message = 'Not Implemented';
      let result  = {'params': JSON.stringify(request.params), success: false, message: message};
      const table = gpu_cache.getDataframe(request.params.table);
      if (table == undefined) {
        result.message = 'Table not found';
        reply.code(404).send(result);
      } else {
        const writer = RecordBatchStreamWriter.writeAll(table.toArrow());
        reply.code(200).send(writer.toNodeStream());
      }
    }
  });
}
