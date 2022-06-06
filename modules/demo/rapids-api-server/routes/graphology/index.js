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
  fastify.get('/', async function(request, reply) {
    return {
      graphology: {
        description: 'The graphology api provides GPU acceleration of graphology datasets.',
        schema: {
          read_json: {
            filename: 'A URI to a graphology json dataset file.',
            returns: 'Result OK/Not Found/Fail'
          },
          read_large_demo: {
            filename:
              'A URI to a graphology json dataset file matching the examples/large-demos spec.',
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
      let message = 'Unknown error';
      let result  = {'params': JSON.stringify(request.query), success: false, message: message};

      console.log(result);
      if (request.query.filename.search('http') != -1) {
        message = 'Remote files not supported yet.'
        console.log(result);
      } else {
        message = 'File is not remote';
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
            result.success = true;
            message        = 'Successfully parsed json file onto GPU.';
            result.message = message;
            try {
              const graphology = gpu_cache.readLargeGraphDemo(path);
              gpu_cache.setDataframe('nodes', graphology['nodes']);
              gpu_cache.setDataframe('edges', graphology['edges']);
              gpu_cache.setDataframe('options', graphology['options']);
              reply.code(200).send(result);
            } catch (e) {
              message        = 'Exception loading dataset onto gpu.';
              result.message = message;
              reply.code(500).send(result);
            }
          }
        } catch (e) {
          message        = 'Exception reading file.';
          result.message = e;
          reply.code(500).send(result);
        };
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
        } catch (e) {
          message        = 'Exception reading file.';
          result.message = e;
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
      let message = 'Error';
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

  fastify.route({
    method: 'GET',
    url: '/get_table/:table',
    schema: {querystring: {table: {type: 'string'}}},
    handler: async (request, reply) => {
      let message = 'Error';
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

  fastify.route({
    method: 'GET',
    url: '/nodes/bounds',
    handler: async (request, reply) => {
      let message = 'Error';
      let result  = {success: false, message: message};
      const df    = gpu_cache.getDataframe('nodes');
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
      const df    = gpu_cache.getDataframe('nodes');
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
        const ratio = () => {
          const [xMin, xMax] = x.minmax();
          const [yMin, yMax] = y.minmax();
          Math.max(xMax - xMin, yMax - yMin);
        };
        const dX = x.minmax().reduce((min, max) => max + min, 0) / 2.0;
        const dY = y.minmax().reduce((min, max) => max + min, 0) / 2.0;
        x        = x.add(-1.0 * dX).mul(1.0 / ratio).add(0.5);
        y        = y.add(-1.0 * dY).mul(1.0 / ratio).add(0.5);
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
      const df    = gpu_cache.getDataframe('nodes');
      const edges = gpu_cache.getDataframe('edges');
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
}
