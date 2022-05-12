'use strict'

const {Series} = require('@rapidsai/cudf');

module.exports = async function(fastify, opts) {
  fastify.get('/', async function(request, reply) {
    const x = Series.new([1, 2, 3]);
    return { root: true, data: x.toArray() }
  });
  fastify.get('/hello', async function(request, reply) { return 'hello'; });
  fastify.route({
    method: 'GET',
    url: '/load-graphology-json',
    schema: {
      querystring: {filename: {type: 'string'}},

      response: {
        200: {
          type: 'object',
          properties:
            {success: {type: 'boolean'}, message: {type: 'string'}, params: {type: 'string'}}
        }
      }
    },
    handler: async (request, reply) => {
      return { 'params': request.query.filename, success: false, message: 'Not implemented.' }
    }
  });
}
