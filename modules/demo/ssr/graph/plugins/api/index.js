// Copyright (c) 2021, NVIDIA CORPORATION.
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

const {graphs, clients}         = require('../graph');
const fs                        = require('fs')
const util                      = require('util')
const {pipeline}                = require('stream')
const pump                      = util.promisify(pipeline)
const glob                      = require('glob');
const {Graph}                   = require('@rapidsai/cugraph');
const {loadEdges, loadNodes}    = require('../graph/loader');
const {RecordBatchStreamWriter} = require('apache-arrow');
const path                      = require('path');
const {readDataFrame, getNodesForGraph, getEdgesForGraph, getPaginatedRows} = require('./utils');

module.exports = function(fastify, opts, done) {
  fastify.addHook('preValidation', (request, reply, done) => {
    // handle upload validation after reading request.file() in the route function itself
    if (request.url == '/datasets/upload') {
      done();
    } else {
      request.query.id =
        (request.method == 'POST') ? `${request.body.id}:video` : `${request.query.id}:video`;
      if (request.query.id in fastify[clients]) {
        done();
      } else {
        reply.code(500).send('client handshake not established');
      }
    }
  });

  async function renderGraph(id, data) {
    const asDeviceMemory = (buf) => new (buf[Symbol.species])(buf);
    const src                    = data.edges.dataframe.get(data.edges.src);
    const dst                    = data.edges.dataframe.get(data.edges.dst);
    const graph                  = Graph.fromEdgeList(src, dst, {directedEdges: true});
    fastify[graphs][id]          = {
      refCount: 0,
      nodes: await getNodesForGraph(asDeviceMemory, data.nodes, graph.numNodes),
      edges: await getEdgesForGraph(asDeviceMemory, data.edges),
      graph: graph,
    };

    ++fastify[graphs][id].refCount;

    return {
      gravity: 0.0,
      linLogMode: false,
      scalingRatio: 5.0,
      barnesHutTheta: 0.0,
      jitterTolerance: 0.05,
      strongGravityMode: false,
      outboundAttraction: false,
      graph: fastify[graphs][id].graph,
      nodes: {
        ...fastify[graphs][id].nodes,
        length: fastify[graphs][id].graph.numNodes,
      },
      edges: {
        ...fastify[graphs][id].edges,
        length: fastify[graphs][id].graph.numEdges,
      },
    };
  }

  fastify.post('/datasets/upload', async function(req, reply) {
    const data = await req.file();
    const id   = `${data.fields.id.value}:video`;
    if (id in fastify[clients]) {
      const basePath = `${__dirname}/../../data/`;
      const filepath = path.join(basePath, data.filename);
      const target   = fs.createWriteStream(filepath);
      try {
        await pump(data.file, target);
      } catch (err) { console.log(err); }
      reply.send();
    } else {
      reply.code(500).send('client handshake not established');
    }
  });

  fastify.get('/datasets', async (request, reply) => {
    glob(`*.{csv,parquet}`,
         {cwd: `${__dirname}/../../data/`},
         (er, files) => { reply.send(JSON.stringify(files.concat(['defaultExample']))); });
  });

  fastify.post('/dataframe/load', async (request, reply) => {
    const filePath = `${__dirname}/../../data/`
    if (fs.existsSync(`${filePath}${request.body.nodes}`) &&
        fs.existsSync(`${filePath}${request.body.edges}`)) {
      fastify[clients][request.query.id].data.nodes.dataframe =
        await readDataFrame(`${filePath}${request.body.nodes}`);

      fastify[clients][request.query.id].data.edges.dataframe =
        await readDataFrame(`${filePath}${request.body.edges}`);
    }
    else {
      fastify[clients][request.query.id].data.nodes.dataframe = await loadNodes();
      fastify[clients][request.query.id].data.edges.dataframe = await loadEdges();
    }
    if (fastify[clients][request.query.id].data.nodes.dataframe.numRows == 0) {
      reply.code(500).send('no dataframe loaded');
    }
    reply.send(JSON.stringify({
      'nodes': fastify[clients][request.query.id].data.nodes.dataframe.numRows,
      'edges': fastify[clients][request.query.id].data.edges.dataframe.numRows
    }));
  })

  fastify.get('/dataframe/columnNames/read', async (request, reply) => {
    reply.send(JSON.stringify({
      nodesParams: fastify[clients][request.query.id].data.nodes.dataframe.names.concat([null]),
      edgesParams: fastify[clients][request.query.id].data.edges.dataframe.names.concat([null])
    }));
  });

  fastify.post('/dataframe/columnNames/update', async (request, reply) => {
    try {
      Object.assign(fastify[clients][request.query.id].data.nodes, request.body.nodes);
      Object.assign(fastify[clients][request.query.id].data.edges, request.body.edges);
      reply.code(200).send('successfully updated columnNames');
    } catch (err) { reply.code(500).send(err); }
  });

  fastify.post('/graph/render', async (request, reply) => {
    try {
      fastify[clients][request.query.id].graph =
        await renderGraph('default', fastify[clients][request.query.id].data);
      reply.code(200).send('successfully rendered graph');
    } catch (err) { reply.code(500).send(err); }
  })

  fastify.get('/dataframe/read', async (request, reply) => {
    try {
      const pageIndex = parseInt(request.query.pageIndex);
      const pageSize  = parseInt(request.query.pageSize);
      const dataframe = request.query.dataframe;  //{'nodes', 'edges'}
      const [arrowTable, numRows] =
        await getPaginatedRows(fastify[clients][request.query.id].data[dataframe].dataframe,
                               pageIndex,
                               pageSize,
                               fastify[clients][request.query.id].state.selectedInfo[dataframe]);

      arrowTable.schema.metadata.set('numRows', numRows);
      RecordBatchStreamWriter.writeAll(arrowTable).pipe(reply.stream());
    } catch (err) {
      request.log.error({err}, '/run_query error');
      reply.code(500).send(err);
    }
  });

  done();
}
