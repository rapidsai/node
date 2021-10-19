const {graphs, clients}          = require('../graph');
const fs                         = require('fs')
const util                       = require('util')
const {pipeline}                 = require('stream')
const pump                       = util.promisify(pipeline)
var glob                         = require('glob');
const {Float32Buffer}            = require('@rapidsai/cuda');
const {GraphCOO}                 = require('@rapidsai/cugraph');
const {DataFrame, Series, Int32} = require('@rapidsai/cudf');
const {loadEdges, loadNodes}     = require('../graph/loader');
const {RecordBatchStreamWriter}  = require('apache-arrow');

function readDataFrame(path) {
  if (path.indexOf('.csv', path.length - 4) !== -1) {
    // csv file
    return DataFrame.readCSV({sources: [path], header: 0, sourceType: 'files'});

  } else if (path.indexOf('.parquet', path.length - 8) !== -1) {
    // csv file
    return DataFrame.readParquet({sources: [path]});
  }
  // if (df.names.includes('Unnamed: 0')) { df = df.cast({'Unnamed: 0': new Uint32}); }
  return new DataFrame({});
}

async function getNodesForGraph(asDeviceMemory, nodes, numNodes) {
  let nodesRes = {};
  const pos    = new Float32Buffer(Array.from(
    {length: numNodes * 2},
    () => Math.random() * 1000 * (Math.random() < 0.5 ? -1 : 1),
    ));

  if (nodes.x in nodes.dataframe.names) {
    nodesRes.nodeXPositions = asDeviceMemory(nodes.dataframe.get(node.x).data);
  } else {
    nodesRes.nodeXPositions = pos.subarray(0, pos.length / 2);
  }
  if (nodes.y in nodes.dataframe.names) {
    nodesRes.nodeYPositions = asDeviceMemory(nodes.dataframe.get(node.y).data);
  } else {
    nodesRes.nodeYPositions = pos.subarray(pos.length / 2);
  }
  if (nodes.dataframe.names.includes(nodes.size)) {
    nodesRes.nodeRadius = asDeviceMemory(nodes.dataframe.get(nodes.size).data);
  }
  if (nodes.dataframe.names.includes(nodes.color)) {
    nodesRes.nodeFillColors = asDeviceMemory(nodes.dataframe.get(nodes.color).data);
  }
  if (nodes.dataframe.names.includes(nodes.id)) {
    nodesRes.nodeElementIndices = asDeviceMemory(nodes.dataframe.get(nodes.id).data);
  }
  return nodesRes;
}

async function getEdgesForGraph(asDeviceMemory, edges) {
  let edgesRes = {};

  if (edges.dataframe.names.includes(edges.color)) {
    edgesRes.edgeColors = asDeviceMemory(edges.dataframe.get(edges.color).data);
  } else {
    edgesRes.edgeColors = asDeviceMemory(
      Series
        .sequence(
          {type: new Uint64, size: edges.dataframe.numRows, init: 18443486512814075489n, step: 0})
        .data);
  }
  if (edges.dataframe.names.includes(edges.id)) {
    edgesRes.edgeList = asDeviceMemory(edges.dataframe.get(edges.id).data);
  }
  if (edges.dataframe.names.includes(edges.bundle)) {
    edgesRes.edgeBundles = asDeviceMemory(edges.dataframe.get(edges.bundle).data);
  }
  return edgesRes;
}

async function getPaginatedRows(df, pageIndex = 1, pageSize = 400, selected = []) {
  console.log(selected);
  if (selected.length == 0) {
    return [df.head(pageIndex * pageSize).tail(pageSize).toArrow(), df.numRows];
  }
  const selectedSeries = Series.new({type: new Int32, data: selected});
  let dfUpdated        = df.gather(selectedSeries);
  return [dfUpdated.head(pageIndex * pageSize).tail(pageSize).toArrow(), dfUpdated.numRows];
}

module.exports = function(fastify, opts, done) {
  // fastify.addHook('preValidation', (request, reply, done) => {
  //   console.log('this is executed', request);
  //   done()
  // });

  async function loadGraph(id, data) {
    if (!(id in fastify[graphs])) {
      const asDeviceMemory = (buf) => new (buf[Symbol.species])(buf);
      const src                    = data.edges.dataframe.get(data.edges.src);
      const dst                    = data.edges.dataframe.get(data.edges.dst);
      const graph                  = new GraphCOO(src._col, dst._col, {directedEdges: true});
      fastify[graphs][id]          = {
        refCount: 0,
        nodes: await getNodesForGraph(asDeviceMemory, data.nodes, graph.numNodes()),
        edges: await getEdgesForGraph(asDeviceMemory, data.edges),
        graph: graph,
      };
    }

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
        length: fastify[graphs][id].graph.numNodes(),
      },
      edges: {
        ...fastify[graphs][id].edges,
        length: fastify[graphs][id].graph.numEdges(),
      },
    };
  }

  fastify.get('/getIDValue', async (request, reply) => {
    console.log(fastify[clients][request.query.id + ':video']);
    reply.send(fastify[clients][request.query.id + ':video'].graph.dataframes[0].numRows);
  });

  fastify.post('/uploadFile', async function(req, reply) {
    const data     = await req.file();
    const basePath = `${__dirname}/../../data/`;
    if (!fs.existsSync(basePath)) { fs.mkdirSync(basePath); }
    const filepath = `${basePath}${data.filename}`;
    const target = fs.createWriteStream(filepath);
    try {
      await pump(data.file, target);
      console.log('success');
    } catch (err) { console.log(err); }
    reply.send()
  });

  fastify.get('/getFileNames', async (request, reply) => {
    if (`${request.query.id}:video` in fastify[clients]) {
      glob(`*.{csv,parquet}`,
           {cwd: `${__dirname}/../../data/`},
           (er, files) => { reply.send(JSON.stringify(files.concat(['defaultExample']))); });
    } else {
      reply.code(500).send('client handshake not established');
    }
  });

  fastify.get('/loadOnGPU', async (request, reply) => {
    const id       = `${request.query.id}:video`;
    const filePath = `${__dirname}/../../data/`
    if (id in fastify[clients]) {
      if (fs.existsSync(`${filePath}${request.query.nodes}`) &&
          fs.existsSync(`${filePath}${request.query.edges}`)) {
        fastify[clients][id].data.nodes.dataframe =
          await readDataFrame(`${filePath}${request.query.nodes}`);

        fastify[clients][id].data.edges.dataframe =
          await readDataFrame(`${filePath}${request.query.edges}`);
      } else {
        fastify[clients][id].data.nodes.dataframe = await loadNodes();
        fastify[clients][id].data.edges.dataframe = await loadEdges();
      }
      if (fastify[clients][id].data.nodes.dataframe.numRows == 0) {
        reply.code(500).send('no dataframe loaded');
      }
      reply.send(JSON.stringify({
        'nodes': fastify[clients][id].data.nodes.dataframe.numRows,
        'edges': fastify[clients][id].data.edges.dataframe.numRows
      }));
    }
    else {
      reply.code(500).send('client handshake not established');
    }
  })

  fastify.get('/fetchDFParameters', async (request, reply) => {
    const id = `${request.query.id}:video`;
    if (id in fastify[clients]) {
      reply.send(JSON.stringify({
        nodesParams: fastify[clients][id].data.nodes.dataframe.names.concat([null]),
        edgesParams: fastify[clients][id].data.edges.dataframe.names.concat([null])
      }));
    } else {
      reply.code(500).send('client handshake not established');
    }
  });

  fastify.post('/updateRenderColumns', async (request, reply) => {
    const id = `${request.body.id}:video`;
    if (id in fastify[clients]) {
      Object.assign(fastify[clients][id].data.nodes, request.body.nodes);
      Object.assign(fastify[clients][id].data.edges, request.body.edges);
      fastify[clients][id].graph = await loadGraph('default', fastify[clients][id].data);
      reply.code(200).send();
    } else {
      reply.code(500).send('client handshake not established');
    }
  });

  fastify.post('/fetchPaginatedData', async (request, reply) => {
    const id = `${request.body.id}:video`;
    if (id in fastify[clients]) {
      const pageIndex = request.body.pageIndex;
      const pageSize  = request.body.pageSize;
      const dataframe = request.body.dataframe;  //{'nodes', 'edges'}
      const [res, numRows] =
        await getPaginatedRows(fastify[clients][id].data[dataframe].dataframe,
                               pageIndex,
                               pageSize,
                               fastify[clients][id].state.selectedInfo[dataframe]);

      reply.send(JSON.stringify({page: res.toArray(), numRows: numRows}));
      // try {
      //   // RecordBatchStreamWriter.writeAll(res).pipe(reply.stream());
      // } catch (err) {
      //   request.log.error({err}, '/run_query error');
      //   reply.code(500).send(err);
      // }
    } else {
      reply.code(500).send('client handshake not established');
    }
  });

  done();
}
