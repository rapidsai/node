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

const Fs = require('fs');
const {Utf8String, Int32, Uint32, Float32, DataFrame, StringSeries, Series, Float64} =
  require('@rapidsai/cudf');
const {RecordBatchStreamWriter, Field, Vector, List, Table} = require('apache-arrow');
const Path                                                  = require('path');
const {promisify}                                           = require('util');
const Stat                                                  = promisify(Fs.stat);
const fastifyCors                                           = require('@fastify/cors');
const fastify                                               = require('fastify');

const arrowPlugin = require('fastify-arrow');
const gpu_cache   = require('../../util/gpu_cache.js');

module.exports = async function(fastify, opts) {
  fastify.register(arrowPlugin);
  fastify.register(fastifyCors, {origin: 'http://localhost:3002'});
  fastify.decorate('setDataframe', gpu_cache.setDataframe);
  fastify.decorate('getDataframe', gpu_cache.getDataframe);
  fastify.decorate('listDataframes', gpu_cache.listDataframes);
  fastify.decorate('readGraphology', gpu_cache.readGraphology);
  fastify.decorate('readLargeGraphDemo', gpu_cache.readLargeGraphDemo);
  fastify.decorate('clearDataFrames', gpu_cache.clearDataframes);
  fastify.get('/', async function(request, reply) {
    return {
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
    }
  });

  fastify.route({
    method: 'POST',
    url: '/read_large_demo',
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
      const path = file.path;
      try {
        const graphology = await fastify.readLargeGraphDemo(path);
        await fastify.setDataframe('nodes', graphology['nodes']);
        await fastify.setDataframe('edges', graphology['edges']);
        await fastify.setDataframe('options', graphology['options']);
        result.message = 'Ok';
        reply.code(200).send(result);
      } catch (e) {
        result.message = 'Exception loading dataset onto gpu.';
        reply.code(500).send(result);
      }
    }
  });

  fastify.route({
    method: 'POST',
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
      const query = request.query;
      let result  = {
        'params': JSON.stringify(query),
        success: false,
        message: 'Finding file.',
        statusCode: 200
      };
      try {
        let path = undefined;
        if (query.filename === undefined) {
          result.message    = 'Parameter filename is required';
          result.statusCode = 400;
        } else if (query.filename.search('http') != -1) {
          result.message    = 'Remote files not supported yet.';
          result.statusCode = 406;
        } else {
          result.message = 'File is not remote';
          // does the file exist?
          const path = Path.join(__dirname, query.filename);
          console.log('Does the file exist?');
          const stats    = await Stat(path);
          const message  = 'File is available';
          result.message = message;
          result.success = true;
          try {
            const graphology = await fastify.readGraphology(path);
            await fastify.setDataframe('nodes', graphology['nodes']);
            await fastify.setDataframe('edges', graphology['edges']);
            await fastify.setDataframe('clusters', graphology['clusters']);
            await fastify.setDataframe('tags', graphology['tags']);
            result.message = 'File read onto GPU.';
          } catch (e) {
            result.success    = false;
            result.message    = e;
            result.statusCode = 500;
          }
        }
      } catch (e) {
        result.success    = false;
        result.message    = e.message;
        result.statusCode = 404;
      };
      await reply.code(result.statusCode).send(result);
    }
  });

  fastify.route({
    method: 'GET',
    url: '/list_tables',
    handler: async (request, reply) => {
      let message = 'Error';
      let result  = {success: false, message: message};
      const list  = await fastify.listDataframes();
      return list;
    }
  });

  fastify.route({
    method: 'GET',
    url: '/get_column/:table/:column',
    schema: {querystring: {table: {type: 'string'}, 'column': {type: 'string'}}},
    handler: async (request, reply) => {
      let message = 'Error';
      let result  = {'params': JSON.stringify(request.params), success: false, message: message};
      const table = await fastify.getDataframe(request.params.table);
      if (table == undefined) {
        result.message = 'Table not found';
        reply.code(404).send(result);
      } else {
        const writer = RecordBatchStreamWriter.writeAll(table.toArrow());
        reply.code(200).send(writer.toNodeStream());
      }
    }
  });

  fastify.route({
    method: 'GET',
    url: '/get_table/:table',
    schema: {querystring: {table: {type: 'string'}}},
    handler: async (request, reply) => {
      let message = 'Error';
      let result  = {'params': JSON.stringify(request.params), success: false, message: message};
      const table = await fastify.getDataframe(request.params.table);
      if (table == undefined) {
        result.message = 'Table not found';
        reply.code(404).send(result);
      } else {
        const writer = RecordBatchStreamWriter.writeAll(table.toArrow());
        reply.code(200).send(writer.toNodeStream());
      }
    }
  });

  fastify.route({
    method: 'GET',
    url: '/nodes/bounds',
    handler: async (request, reply) => {
      let message = 'Error';
      let result  = {success: false, message: message};
      const df    = await fastify.getDataframe('nodes');
      if (df == undefined) {
        result.message = 'Table not found';
        reply.code(404).send(result);
      } else {
        // compute xmin, xmax, ymin, ymax
        const x = df.get('x');
        const y = df.get('y');

        result.bounds =
          {xmin: x._col.min(), xmax: x._col.max(), ymin: y._col.min(), ymax: y._col.max()};
        result.message = 'Success';
        result.success = true;
        reply.code(200).send(result);
      }
    }
  });

  fastify.route({
    method: 'GET',
    url: '/nodes/',
    handler: async (request, reply) => {
      let message = 'Error';
      let result  = {success: false, message: message};
      const df    = await fastify.getDataframe('nodes');
      if (df == undefined) {
        result.message = 'Table not found';
        reply.code(404).send(result);
      } else {
        // tile x, y, size, color
        let tiled       = Series.sequence({type: new Float32, init: 0.0, size: (4 * df.numRows)});
        let base_offset = Series.sequence({type: new Int32, init: 0.0, size: df.numRows}).mul(4);
        //
        // Duplicatin the sigma.j createNormalizationFunction here because there's no other way
        // to let the Graph object compute it.
        //
        let x       = df.get('x');
        let y       = df.get('y');
        let color   = df.get('color');
        const ratio = (() => {
          const [xMin, xMax] = x.minmax();
          const [yMin, yMax] = y.minmax();
          return Math.max(xMax - xMin, yMax - yMin);
        })();
        const dX    = x.minmax().reduce((min, max) => max + min, 0) / 2.0;
        const dY    = y.minmax().reduce((min, max) => max + min, 0) / 2.0;
        x           = x.add(-1.0 * dX).mul(1.0 / ratio).add(0.5);
        y           = y.add(-1.0 * dY).mul(1.0 / ratio).add(0.5);
        // done with createNormalizationFunction
        tiled = tiled.scatter(x, base_offset.cast(new Int32));
        tiled = tiled.scatter(y, base_offset.add(1).cast(new Int32));
        tiled = tiled.scatter(df.get('size').mul(2), base_offset.add(2).cast(new Int32));
        color = color.hexToIntegers(new Uint32).bitwiseOr(0xff000000);
        // color = Series.sequence({size: color.length, type: new Int32, init: 0xff0000ff, step:
        // 0});
        tiled        = tiled.scatter(color.view(new Float32), base_offset.add(3).cast(new Int32));
        const writer = RecordBatchStreamWriter.writeAll(new DataFrame({nodes: tiled}).toArrow());
        reply.code(200).send(writer.toNodeStream());
      }
    }
  });

  fastify.route({
    method: 'GET',
    url: '/edges/',
    handler: async (request, reply) => {
      let message = 'Error';
      let result  = {success: false, message: message};
      const df    = await fastify.getDataframe('nodes');
      const edges = await fastify.getDataframe('edges');
      if (df == undefined) {
        result.message = 'Table not found';
        reply.code(404).send(result);
      } else {
        // tile x, y, size, color
        let tiled = Series.sequence({type: new Float32, init: 0.0, size: (6 * edges.numRows)});
        let base_offset =
          Series.sequence({type: new Int32, init: 0.0, size: edges.numRows})._col.mul(3);
        //
        // Duplicatin the sigma.j createNormalizationFunction here because there's no other way
        // to let the Graph object compute it.
        //
        let source = edges.get('source');
        let target = edges.get('target');
        let x      = df.get('x');
        let y      = df.get('y');
        const ratio =
          Series.new([x._col.max() - x._col.min(), y._col.max() - y._col.min()])._col.max();
        const dX          = (x._col.max() + x._col.min()) / 2.0;
        const dY          = (y._col.max() + y._col.min()) / 2.0;
        x                 = x._col.add(-1.0 * dX).mul(1.0 / ratio).add(0.5);
        y                 = y._col.add(-1.0 * dY).mul(1.0 / ratio).add(0.5);
        const source_xmap = x.gather(source._col.cast(new Int32));
        const source_ymap = y.gather(source._col.cast(new Int32));
        const target_xmap = x.gather(target._col.cast(new Int32));
        const target_ymap = y.gather(target._col.cast(new Int32));
        const color       = Series.new(['#999'])
                        .hexToIntegers(new Int32)
                        .bitwiseOr(0xff000000)
                        .view(new Float32)
                        .toArray()[0];
        tiled =
          tiled.scatter(Series.new(source_xmap), Series.new(base_offset.mul(2)).cast(new Int32));
        tiled        = tiled.scatter(Series.new(source_ymap),
                              Series.new(base_offset.mul(2).add(1)).cast(new Int32));
        tiled        = tiled.scatter(color, Series.new(base_offset.mul(2).add(2).cast(new Int32)));
        tiled        = tiled.scatter(Series.new(target_xmap),
                              Series.new(base_offset.mul(2).add(3)).cast(new Int32));
        tiled        = tiled.scatter(Series.new(target_ymap),
                              Series.new(base_offset.mul(2).add(4)).cast(new Int32));
        tiled        = tiled.scatter(color, Series.new(base_offset.mul(2).add(5).cast(new Int32)));
        const writer = RecordBatchStreamWriter.writeAll(new DataFrame({edges: tiled}).toArrow());
        reply.code(200).send(writer.toNodeStream());
      }
    }
  });

  fastify.route({
    method: 'POST',
    url: '/release',
    handler: async (request, reply) => {
      await fastify.clearDataFrames();
      reply.code(200).send({message: 'OK'})
    }
  });
}
