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

const {Int32, Float32, DataFrame, Series} = require('@rapidsai/cudf');
const {RecordBatchStreamWriter}           = require('apache-arrow');
const fastify                             = require('fastify')({logger: {level: 'debug'}});
const fastifyCors                         = require('@fastify/cors');

const arrowPlugin = require('fastify-arrow');
const gpu_cache   = require('../../util/gpu_cache.js');
const root_schema = require('../../util/schema.js');

module.exports = async function(fastify, opts) {
  fastify.register(fastifyCors, {origin: '*'});
  fastify.register(arrowPlugin);
  fastify.decorate('setDataframe', gpu_cache.setDataframe);
  fastify.decorate('getDataframe', gpu_cache.getDataframe);
  fastify.decorate('readCSV', gpu_cache.readCSV);

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

  fastify.get('/', {...get_schema, handler: () => root_schema['particles']});
  fastify.post('/', {...get_schema, handler: () => root_schema['particles']});

  const handler = async (request, reply) => {
    let message = 'Error';
    let result  = {'params': JSON.stringify(request.params), success: false, message: message};
    console.log(result);
    const table = await fastify.getDataframe(request.params.table);
    if (table == undefined) {
      result.message = 'Table not found';
      await reply.code(404).send(result);
    } else {
      try {
        let x = undefined;
        let y = undefined;
        if (request.params.xmin != undefined && request.params.xmax != undefined &&
            request.params.ymin != undefined && request.params.ymax != undefined) {
          const x_unfiltered = table.get('Longitude');
          const x_gt         = x_unfiltered._col.gt(parseFloat(request.params.xmin));
          const x_lt         = x_unfiltered._col.lt(parseFloat(request.params.xmax));
          const y_unfiltered = table.get('Latitude');
          const y_gt         = y_unfiltered._col.gt(parseFloat(request.params.ymin));
          const y_lt         = y_unfiltered._col.lt(parseFloat(request.params.ymax));
          const x_y          = x_lt.bitwiseAnd(x_gt).bitwiseAnd(y_lt).bitwiseAnd(y_gt);
          x                  = x_unfiltered.filter(Series.new(x_y));
          y                  = y_unfiltered.filter(Series.new(x_y));
        } else {
          x = table.get('Longitude');
          y = table.get('Latitude');
        }
        // const state = table.get('State');
        // const zip   = table.get('Zip_Code')
        // Produce r,g,b from state
        const color_map = [
          {'r': 0, 'g': 0, 'b': 0},
          {'r': 255, 'g': 0, 'b': 0},
          {'r': 0, 'g': 255, 'b': 0},
          {'r': 255, 'g': 255, 'b': 0},
          {'r': 0, 'g': 0, 'b': 255},
          {'r': 255, 'g': 0, 'b': 255},
          {'r': 0, 'g': 255, 'b': 255},
          {'r': 255, 'g': 255, 'b': 255}
        ];
        // TODO: convert state to color by state index
        const r = Series.sequence({type: new Int32, init: 255.0, size: x.length}).fill(0);
        const g = Series.sequence({type: new Int32, init: 255.0, size: x.length}).fill(0);
        const b = Series.sequence({type: new Int32, init: 255.0, size: x.length}).fill(0);

        // Map x, y, r, g, b to offsets for client display
        let tiled       = Series.sequence({type: new Float32, init: 0.0, size: (7 * x.length)});
        let base_offset = Series.sequence({type: new Int32, init: 0.0, size: x.length}).mul(7);
        tiled           = tiled.scatter(x, base_offset.cast(new Int32));
        x.dispose();
        tiled = tiled.scatter(y, base_offset.add(1).cast(new Int32));
        y.dispose();
        tiled = tiled.scatter(1.0, base_offset.add(2).cast(new Int32));
        tiled = tiled.scatter(1.0, base_offset.add(3).cast(new Int32));
        tiled = tiled.scatter(r, base_offset.add(4).cast(new Int32));
        r.dispose();
        tiled = tiled.scatter(g, base_offset.add(5).cast(new Int32));
        g.dispose();
        tiled = tiled.scatter(b, base_offset.add(6).cast(new Int32));
        b.dispose();
        const result = new DataFrame({'gpu_buffer': tiled});
        const writer = RecordBatchStreamWriter.writeAll(result.toArrow());
        await reply.code(200).send(writer.toNodeStream());
      } catch (e) {
        result.message = e.message;
        if (e.message.search('Unknown column name') != -1) {
          result.message = {
            error: result.message,
            message:
              'Imported CSV file must contain four columns: State, Zip_Code, Longitude, and Latitude'
          };
          await reply.code(500).send(result);
        } else {
          await reply.code(500).send(result);
        }
      }
    }
  };

  fastify.route({
    method: 'GET',
    url: '/get_shader_column/:table/:xmin/:xmax/:ymin/:ymax',
    schema: {
      querystring: {
        table: {type: 'string'},
        xmin: {type: 'number'},
        xmax: {type: 'number'},
        ymin: {type: 'number'},
        ymax: {type: 'number'}
      }
    },
    handler: handler
  });
  fastify.route({
    method: 'GET',
    url: '/get_shader_column/:table',
    schema: {querystring: {table: {type: 'string'}}},
    handler: handler
  });
};
