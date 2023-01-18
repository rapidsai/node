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
    READ_CSV_URL: '/gpu/DataFrame/readCSV',
    readCsvOptions: (filename) => {
      return {
        method: 'POST', headers: JSON_CORS_HEADERS, body: JSON.stringify({filename: filename})
      }
    }
  },
  create_quadtree: {
    CREATE_QUADTREE_URL: '/quadtree/create/',
    createQuadtreeOptions: (xAxisName, yAxisName) => {
      return {
        method: 'POST', headers: JSON_CORS_HEADERS,
          body: JSON.stringify({xAxisName: xAxisName, yAxisName: yAxisName})
      }
    },
  },
  set_polygon: {
    SET_POLYGON_URL: '/quadtree/set_polygons',
    setPolygonOptions: (name, polygonPoints) => {
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
  get_points: {
    GET_POINTS_URL: '/quadtree/get_points',
    GET_POINTS_OPTIONS: {method: 'GET', headers: JSON_CORS_HEADERS}
  }
}

export const readCsv = async (filename) => {
  /*
    readCsv reads the csv file to the server.
    */
  const result     = await fetch(config.SERVER + ':' + config.PORT + config.read_csv.READ_CSV_URL,
                             config.read_csv.readCsvOptions(filename));
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
  const result = await fetch(
    config.SERVER + ':' + config.PORT + config.create_quadtree.CREATE_QUADTREE_URL + csvName,
    config.create_quadtree.createQuadtreeOptions(axisNames.x, axisNames.y));
  const resultJson = await result.json()
  return resultJson.params.quadtree;
};

export const setPolygon = async (name, polygonPoints) => {
  /* setPolygon sets the polygon to be used for the quadtree. */
  const result = await fetch(config.SERVER + ':' + config.PORT + config.set_polygon.SET_POLYGON_URL,
                             config.set_polygon.setPolygonOptions(name, polygonPoints));
  const resultJson = await result.json();
  return resultJson.params.name;
};

export const getQuadtreePoints =
  /* getQuadtreePoints gets the points from the quadtree. */
  async (quadtreeName, polygonName, n) => {
    let path = config.SERVER + ':' + config.PORT + config.get_points.GET_POINTS_URL + '/' +
               quadtreeName + '/' + polygonName;
    path += n != undefined ? '/' + n : '';
    const start        = Date.now();
    const remotePoints = await fetch(path, config.get_points.GET_POINTS_OPTIONS);
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
