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

const Fs                                                               = require('fs');
const {Utf8String, Int32, Uint32, Float32, DataFrame, Series, Float64} = require('@rapidsai/cudf');
const {RecordBatchStreamWriter, Field, Vector, List, Table}            = require('apache-arrow');
const Path                                                             = require('path');
const {promisify}                                                      = require('util');
const Stat                                                             = promisify(Fs.stat);
const fastifyCors                                                      = require('@fastify/cors');
const fastify                                                          = require('fastify');
const cudf_api                                                         = require('@rapidsai/cudf');

const arrowPlugin = require('fastify-arrow');
const gpu_cache   = require('../../util/gpu_cache.js');
const root_schema = require('../../util/schema.js');

module.exports = async function(fastify, opts) {
  fastify.register(arrowPlugin);
  fastify.register(fastifyCors, {origin: '*'});
  fastify.decorate('setDataframe', gpu_cache.setDataframe);
  fastify.decorate('getDataframe', gpu_cache.getDataframe);
  fastify.decorate('gpu', gpu_cache);

  const get_handler =
    async (request, reply) => {
    const query = request.query;
    request.log.info('Parsing Query:');
    request.log.info(query);
    request.log.info('Sending query to gpu_cache');
    request.log.info('Updating result');
    request.log.info('Sending cache.tick');
    let result = {
      'params': JSON.stringify(query),
      success: true,
      message: `gpu method:${request.method} placeholder`,
      statusCode: 200
    };
    return result
  }

  const cudf =
    async (route, args) => {
    debugger;
    const evalString   = route.join('.');
    const paramsString = '(' + args + ')';
    eval(evalString + paramsString);
  }

  const cudf_dispatcher =
    async (route, args) => {
    const fn = Function(route[0]);
    eval(route[0])(route, args);
  }

  const post_handler =
    async (request, reply) => {
    request.log.info('Parsing Url:');
    const url = request.url.split('/');
    url.shift();
    url.shift();
    const query = request.query;
    request.log.info('Sending query to gpu_cache');
    cudf_dispatcher(url, query);
    request.log.info('Updating result');
    request.log.info('Sending cache.tick');
    let result = {
      'params': JSON.stringify(query),
      success: true,
      message: `gpu method:${request.method} placeholder`,
      statusCode: 200
    };
    return result
  }

  const get_schema = {
    logLevel: 'debug',
    schema: {
      response: {
        200: {
          type: 'object',
          properties:
            {success: {type: 'boolean'}, message: {type: 'string'}, params: {type: 'string'}}
        }
      }
    }
  };

  fastify.get('/', {...get_schema, handler: get_handler});
  fastify.post('/', {...get_schema, handler: post_handler});
  fastify.get('/*', {...get_schema, handler: get_handler});
  fastify.post('/*', {...get_schema, handler: post_handler});
}
