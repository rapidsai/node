const {DataFrame, Series, Int32, Uint8, Uint32, Uint64} = require('@rapidsai/cudf');
const {Float32Buffer}                                   = require('@rapidsai/cuda');

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
    nodesRes.nodeRadius = asDeviceMemory(nodes.dataframe.get(nodes.size).cast(new Uint8).data);
  }
  if (nodes.dataframe.names.includes(nodes.color)) {
    nodesRes.nodeFillColors =
      asDeviceMemory(nodes.dataframe.get(nodes.color).cast(new Uint32).data);
  }
  if (nodes.dataframe.names.includes(nodes.id)) {
    nodesRes.nodeElementIndices =
      asDeviceMemory(nodes.dataframe.get(nodes.id).cast(new Uint32).data);
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
    edgesRes.edgeList = asDeviceMemory(edges.dataframe.get(edges.id).cast(new Uint64).data);
  }
  if (edges.dataframe.names.includes(edges.bundle)) {
    edgesRes.edgeBundles = asDeviceMemory(edges.dataframe.get(edges.bundle).data);
  }
  return edgesRes;
}

async function getPaginatedRows(df, pageIndex = 0, pageSize = 400, selected = []) {
  if (selected.length != 0) {
    const selectedSeries = Series.new({type: new Int32, data: selected}).unique(true);
    const updatedDF      = df.gather(selectedSeries);
    const idxs           = Series.sequence({
      type: new Int32,
      init: (pageIndex - 1) * pageSize,
      size: Math.min(pageSize, updatedDF.numRows),
      step: 1
    });
    return [updatedDF.gather(idxs).toArrow(), updatedDF.numRows];
  } else {
    const idxs = Series.sequence({
      type: new Int32,
      init: (pageIndex - 1) * pageSize,
      size: Math.min(pageSize, df.numRows),
      step: 1
    });
    return [df.gather(idxs).toArrow(), df.numRows];
  }
}

module.exports = {
  readDataFrame,
  getNodesForGraph,
  getEdgesForGraph,
  getPaginatedRows
}
