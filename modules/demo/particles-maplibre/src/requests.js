/**
 * Copyright (c) 2022 NVIDIA Corporation
 */

const {tableFromIPC} = require('apache-arrow');

const JSON_CORS_HEADERS = {
  'access-control-allow-origin': '*',
  'Content-Type': 'application/json',
  'Access-Control-Allow-Headers': 'Content-Type'
}

const config = {
  SERVER: 'http://localhost',
  PORT: '3010',
  read_csv: {
    URL: '/gpu/DataFrame/readCSV',
    options: (filename) => {
      return {
        method: 'POST', headers: JSON_CORS_HEADERS, body: JSON.stringify({filename: filename})
      }
    }
  },
  create_quadtree: {
    URL: '/quadtree/create/',
    options: (xAxisName, yAxisName) => {
      return {
        method: 'POST', headers: JSON_CORS_HEADERS,
          body: JSON.stringify({xAxisName: xAxisName, yAxisName: yAxisName})
      }
    },
  },
  set_polygon: {
    URL: '/quadtree/set_polygons',
    options: (name, polygonPoints) => {
      return {
        method: 'POST', headers: JSON_CORS_HEADERS, body: JSON.stringify({
          name: name,
          polygon_offset: [0, 1],
          ring_offset: [0, polygonPoints.length],
          points: polygonPoints
        })
      }
    },
  },
  get_points: {URL: '/quadtree/get_points', OPTIONS: {method: 'GET', headers: JSON_CORS_HEADERS}},
  count: {URL: '/quadtree', OPTIONS: {method: 'GET', headers: JSON_CORS_HEADERS}},
  release: {URL: '/gpu/release', OPTIONS: {method: 'POST'}},
}

export const readCsv = async (filename) => {
  /*
    readCsv reads the csv file to the server.
    */
  const result     = await fetch(config.SERVER + ':' + config.PORT + config.read_csv.URL,
                             config.read_csv.options(filename));
  const resultJson = await result.json();
  return resultJson.params.filename;
};

export const createQuadtree = async (csvName, axisNames) => {
  /*
    createQuadtree creates a quadtree from the points in props.points.
    This is used to determine which points are in the current viewport
    and which are not. This is used to determine which points to render
    and which to discard.
    */
  const result =
    await fetch(config.SERVER + ':' + config.PORT + config.create_quadtree.URL + csvName,
                config.create_quadtree.options(axisNames.x, axisNames.y));
  const resultJson = await result.json()
  return resultJson.params.quadtree;
};

export const setPolygon = async (name, polygonPoints) => {
  /* setPolygon sets the polygon to be used for the quadtree. */
  const result     = await fetch(config.SERVER + ':' + config.PORT + config.set_polygon.URL,
                             config.set_polygon.options(name, polygonPoints));
  const resultJson = await result.json();
  return resultJson.params.name;
};

export const getQuadtreePoints =
  /* getQuadtreePoints gets the points from the quadtree. */
  async (quadtreeName, polygonName, n) => {
    let path = config.SERVER + ':' + config.PORT + config.get_points.URL + '/' + quadtreeName +
               '/' + polygonName;
    path += n !== undefined ? '/' + n + '/next' : '';
    const start        = Date.now();
    const remotePoints = await fetch(path, config.get_points.OPTIONS);
    const end          = Date.now();
    console.log('Time to fetch: ' + (end - start));
    if (remotePoints.ok) {
      const arrowTable = await tableFromIPC(remotePoints);
      return arrowTable.getChildAt(0).toArray();
    } else {
      console.log('Unable to fetch');
      console.log(remotePoints);
    }
  }

export const getQuadtreePointCount =
  /* getQuadtreePoints gets the count of a points in a polygon from from the quadtree. */
  async (quadtreeName, polygonName) => {
    let path = config.SERVER + ':' + config.PORT + config.count.URL + '/' + quadtreeName + '/' +
               polygonName + '/count';
    const countResult = await fetch(path, config.count.OPTIONS);
    return await countResult.json();
  }

export const release = async (quadtreeName) => {
  /* release resets the remote server. */
  const result =
    await fetch(config.SERVER + ':' + config.PORT + config.release.URL, config.release.OPTIONS);
  return result;
}
