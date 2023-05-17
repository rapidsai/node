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

const {Float64, Float32, DataFrame, Series} = require('@rapidsai/cudf');
const {RecordBatchStreamWriter}             = require('apache-arrow');
const fastify                               = require('fastify')({logger: {level: 'debug'}});
const fastifyCors                           = require('@fastify/cors');
const cuspatial                             = require('@rapidsai/cuspatial');

const rapids_viewer = require('../../util/rapids-viewer.js');
const arrowPlugin   = require('fastify-arrow');
const gpu_cache     = require('../../util/gpu_cache.js');
const root_schema   = require('../../util/schema.js');

module.exports = async function(fastify, opts) {
  fastify.register(fastifyCors, {origin: '*'});
  fastify.register(arrowPlugin);
  fastify.decorate('cacheObject', gpu_cache.cacheObject);
  fastify.decorate('getData', gpu_cache.getData);
  fastify.decorate('listDataframes', gpu_cache.listDataframes);
  fastify.decorate('rapids_viewer', rapids_viewer);
  fastify.decorate('publicPath', gpu_cache.publicPath);

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

  fastify.get('/', {...get_schema, handler: () => root_schema['rapids-viewer']});
  fastify.post('/', {...get_schema, handler: () => root_schema['rapids-viewer']});

  fastify.route({
    method: 'POST',
    url: '/set_dataframe',
    schema: {
      querystring: {
        dataframe: {type: 'string'},
        xAxisName: {type: 'string'},
        yAxisName: {type: 'string'},
      }
    },
    handler: async (request, reply) => {
      const df = await fastify.getData(request.body.dataframe);
      rapids_viewer.set_df(df, request.body.xAxisName, request.body.yAxisName);
      reply.code(200).send({success: true});
    }
  });

  fastify.route({
    method: 'POST',
    url: '/set_viewport',
    schema: {querystring: {lb: {type: 'array'}, ub: {type: 'array'}}},
    handler: async (request, reply) => {
      const {lb, ub} = request.body;
      rapids_viewer.set_viewport(lb, ub);
      reply.code(200).send({success: true});
    }
  });

  fastify.route({
    method: 'POST',
    url: '/change_budget',
    schema: {querystring: {budget: {type: 'number'}}},
    handler: async (request, reply) => {
      const {budget} = request.body;
      rapids_viewer.change_budget(budget);
      reply.code(200).send({success: true});
    }
  });

  fastify.route({
    method: 'GET',
    url: '/get_n/:n',
    schema: {querystring: {n: {type: 'number'}}},
    handler: async (request, reply) => {
      const {n}    = request.params;
      n_points     = rapids_viewer.get_n(parseInt(n));
      const writer = RecordBatchStreamWriter.writeAll(n_points.toArrow());
      writer.close();
      await reply.code(200).send(writer.toNodeStream());
    }
  });
}
