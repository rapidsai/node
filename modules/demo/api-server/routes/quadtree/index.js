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
const cuspatial                           = require('@rapidsai/cuspatial');

const arrowPlugin = require('fastify-arrow');
const gpu_cache   = require('../../util/gpu_cache.js');
const root_schema = require('../../util/schema.js');

module.exports = async function(fastify, opts) {
  fastify.register(fastifyCors, {origin: '*'});
  fastify.register(arrowPlugin);
  fastify.decorate('cacheObject', gpu_cache.cacheObject);
  fastify.decorate('getData', gpu_cache.getData);
  fastify.decorate('listDataframes', gpu_cache.listDataframes);
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

  fastify.get('/', {...get_schema, handler: () => root_schema['quadtree']});
  fastify.post('/', {...get_schema, handler: () => root_schema['quadtree']});

  fastify.route({
    method: 'POST',
    url: '/create/:table',
    schema: {querystring: {table: {type: 'string'}}},
    handler: async (request, reply) => {
      let message = 'Error';
      let result  = {'params': request.params, success: false, message: message};
      const table = await fastify.getData(request.params.table);
      if (request.body.xAxisName === undefined || request.body.yAxisName === undefined) {
        result.message = 'xAxisName or yAxisName undefined, specify them in POST body.';
        result.code    = 400;
        await reply.code(result.code).send(result);
        return;
      }
      if (table == undefined) {
        result.message = 'Table not found';
        await reply.code(404).send(result);
      } else {
        const [xMin, xMax, yMin, yMax] = [
          parseFloat(table.get(request.body.xAxisName).min()),
          parseFloat(table.get(request.body.xAxisName).max()),
          parseFloat(table.get(request.body.yAxisName).min()),
          parseFloat(table.get(request.body.yAxisName).max()),
        ];
        try {
          const quadtree          = cuspatial.Quadtree.new({
            x: table.get(request.body.xAxisName),
            y: table.get(request.body.yAxisName),
            xMin,
            xMax,
            yMin,
            yMax,
            scale: 0,
            maxDepth: 15,
            minSize: 1e5
          });
          const quadtree_name     = request.params.table + '_quadtree';
          request.params.quadtree = quadtree_name
          const saved             = await fastify.cacheObject(quadtree_name, quadtree);
          result.message          = 'Quadtree created';
          result.success          = true;
          result.statusCode       = 200;
          await reply.code(result.statusCode).send(result);
        } catch (e) {
          result.message    = e;
          result.success    = false;
          result.statusCode = 500;
          await reply.code(result.statusCode).send(result);
        }
      }
    }
  });

  fastify.route({
    method: 'POST',
    url: '/set_polygons_quadtree',
    schema: {
      querystring: {
        name: {type: 'string'},
        polygon_offset: {type: 'array'},
        ring_offset: {type: 'array'},
        points: {type: 'array'}
      }
    },
    handler: async (request, reply) => {
      let message = 'Error';
      let result  = {'params': request.params, success: false, message: message};
      try {
        const polygon_offset = Series.new(new Int32Array(request.body.polygon_offset));
        const ring_offset    = Series.new(new Int32Array(request.body.ring_offset));
        const points         = Series.new(new Float64Array(request.body.points));
        const cached =
          await fastify.cacheObject(request.body.name, {polygon_offset, ring_offset, points});
        result.message    = 'Set polygon ' + request.body.name;
        result.success    = true;
        result.statusCode = 200;
        result.params     = request.body;
        await reply.code(result.statusCode).send(result);
      } catch (e) {
        result.message    = e;
        result.success    = false;
        result.statusCode = 500;
        await reply.code(result.statusCode).send(result);
      }
    }
  });

  fastify.route({
    method: 'GET',
    url: '/get_points/:quadtree/:polygon',
    schema: {querystring: {quadtree: {type: 'string'}, polygon: {type: 'string'}}},
    handler: async (request, reply) => {
      let message = 'Error';
      let result  = {'params': request.params, success: false, message: message};
      try {
        const quadtree = await fastify.getData(request.params.quadtree);
        const {polygon_offset, ring_offset, points} = await fastify.getData(request.params.polygon);
        const data                                  = await fastify.listDataframes();
        const pts                                   = cuspatial.makePoints(
          points.gather(Series.sequence({size: points.length, step: 2, init: 0})),
          points.gather(Series.sequence({size: points.length, step: 2, init: 1})));
        const polylines      = cuspatial.makePolylines(pts, ring_offset);
        const polygons       = cuspatial.makePolygons(polylines, polygon_offset);
        const polyPointPairs = quadtree.pointInPolygon(polygons);
        const resultPoints   = quadtree.points.gather(polyPointPairs.get('point_index'));
        const numPoints      = resultPoints.get('x').length
        let result_col       = Series.sequence({size: numPoints * 2, step: 0, init: 0});
        result_col           = result_col.scatter(resultPoints.get('x'),
                                        Series.sequence({size: numPoints, step: 2, init: 0}));
        result_col           = result_col.scatter(resultPoints.get('y'),
                                        Series.sequence({size: numPoints, step: 2, init: 1}));
        result               = new DataFrame({'points_in_polygon': result_col})
        const writer         = RecordBatchStreamWriter.writeAll(result.toArrow());
        writer.close();
        await reply.code(200).send(writer.toNodeStream());
      } catch (e) {
        result.message    = e;
        result.success    = false;
        result.statusCode = 500;
        await reply.code(result.statusCode).send(result);
      }
    }
  });
}