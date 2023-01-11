/**
 * Copyright (c) 2022 NVIDIA Corporation
 */

import React from 'react';
import ReactDOM from 'react-dom';

import App from './App';
import background from './background';
const {drawParticles, particlesEngine} = require('./points');

const {tableFromIPC}           = require('apache-arrow');
const mat4                     = require('gl-mat4');
const {getScreenToWorldCoords} = require('./matrices');

(async () => {
  /*
    Homogeneous coordinates for projection matrix
    Projecting from one 2d basis to another using control points
    requires 3 3d coordinates in order to capture translation in
    the projection matrix. OpenGL is a 3d system with a 4d
    homogeneous matrix patter. We'll use mat4 to do 4d matrix
    transformations on our 2d points. Throw away the third dimension
    and keep the homogeneous coordinate.

    These will be used in props to align the screen and world coordinate
    systems.
  */
  const worldCoords = [
    -137.3,
    49.2,  // top right
    0,
    1,
    -61.6,
    49.2,  // top left
    0,
    1,
    -61.6,
    25.2,  // bottom left
    0,
    1,
    -137.3,
    25.2,  // bottom right
    0,
    1
  ];
  /*
   Screen coords matching the axes of the above worldCoords. They are used to
   compute the world view to screen projection matrix.

   These are inadequate for aligning the mouse movements correctly with the world movement, but they
   provide enough information to create a viewPort alignment. TODO
   */
  const screenCoords = [
    document.documentElement.clientWidth,
    document.documentElement.clientHeight,
    0,
    1,
    0,
    document.documentElement.clientHeight,
    0,
    1,
    0,
    0,
    0,
    1,
    document.documentElement.clientWidth,
    0,
    0,
    1
  ];

  const props = {
    /*
     All application state except for a few constants
     are defined here. This state must be shared between
     components.
     */
    // Define world coords
    w: () => [...worldCoords],
    // Define screen coords
    s: () => [...screenCoords],
    zoomLevel: 15,
    angle: 0,
    screenWidth: document.documentElement.clientHeight,
    screenHeight: document.documentElement.clientWidth,
    centerX: document.documentElement.clientWidth / 2.0,
    centerY: document.documentElement.clientHeight / 2.0,
    currentWorldCoords: {xmin: undefined, xmax: undefined, ymin: undefined, ymax: undefined},
    fetching: false,
    pointBudget: 250000,
    displayPointCount: 0,
  };

  /*
   Mouse events.
   */
  window.addEventListener('wheel', (event) => {
    /*
    Scroll/zoom event, update props to re-render the world.
    */
    props.centerX      = props.centerX + event.movementX;
    props.centerY      = props.centerY + event.movementY;
    props.screenWidth  = document.documentElement.clientWidth;
    props.screenHeight = document.documentElement.clientHeight;
    const zoom         = event.deltaY > 0 ? -1 : 1;
    props.zoomLevel    = props.zoomLevel + zoom;
    /*
     newWorldCoords defines the bounding in world-coordinates of the current
     view. This is used to update the points using server based viewport culling.
     */
    const newWorldCoords          = getScreenToWorldCoords(props);
    props.currentWorldCoords.xmin = newWorldCoords[0];
    props.currentWorldCoords.xmax = newWorldCoords[8];
    props.currentWorldCoords.ymin = newWorldCoords[9];
    props.currentWorldCoords.ymax = newWorldCoords[1];
    // fetchPoints(csvName, props);
  });
  window.addEventListener('mousedown', (event) => {
    /*
     isHeld prop to track dragging events.
    */
    props.isHeld = true;
  });
  window.addEventListener('mouseup', (event) => {
    /*
     Disable dragging when released.
    */
    props.isHeld = false;
  });
  window.addEventListener('mousemove', (event) => {
    /*
     Update the current "center" of the viewport in order to change
     projection of the points and background. Needs to be updated
     to better track the difference between the screen and the viewport. TODO
     */
    if (props.isHeld) {
      props.centerX        = props.centerX + event.movementX;
      props.centerY        = props.centerY + event.movementY;
      props.screenWidth    = document.documentElement.clientWidth;
      props.screenHeight   = document.documentElement.clientHeight;
      const newWorldCoords = getScreenToWorldCoords(props);
      console.log(newWorldCoords);
      props.currentWorldCoords.xmin = newWorldCoords[0];
      props.currentWorldCoords.xmax = newWorldCoords[8];
      props.currentWorldCoords.ymin = newWorldCoords[9];
      props.currentWorldCoords.ymax = newWorldCoords[1];
      // fetchPoints(csvName, props, newWorldCoords);
    }
  });

  /*
   Client config
   */
  const SERVER           = 'http://localhost';
  const PORT             = '3010';
  const READ_CSV_URL     = '/gpu/DataFrame/readCSV';
  const READ_CSV_OPTIONS = {
    method: 'POST',
    headers: {
      'access-control-allow-origin': '*',
      'Content-Type': 'application/json',
      'Access-Control-Allow-Headers': 'Content-Type'
    },
    /*
     Different data files to choose from for client. This would
     be great to have a widget or text field for entry. TODO
     */
    // body: '"NAD_30m.csv"',
    // body: '"NAD_State_ZIP_LonLat.csv"',
    body: '{"filename": "cartesian_250k.csv"}',
    // body: '{"filename": "NAD_Shuffled_100000.csv"}',
  };

  const createQuadtree = async (csvName, props) => {
    /*
      createQuadtree creates a quadtree from the points in props.points.
      This is used to determine which points are in the current viewport
      and which are not. This is used to determine which points to render
      and which to discard.
      */
    const CREATE_QUADTREE_URL     = '/quadtree/create/';
    const CREATE_QUADTREE_OPTIONS = {
      method: 'POST',
      headers: {
        'access-control-allow-origin': '*',
        'Content-Type': 'application/json',
        'Access-Control-Allow-Headers': 'Content-Type'
      },
      body: JSON.stringify({xAxisName: 'Longitude', yAxisName: 'Latitude'})
    };
    const result =
      await fetch(SERVER + ':' + PORT + CREATE_QUADTREE_URL + csvName, CREATE_QUADTREE_OPTIONS)
    const resultJson = await result.json()
    return resultJson.params.quadtree;
  };

  const setPolygon = async (which, props) => {
    /*
      setPolygon is used to set the polygon for each of the four test quadrants
      */
    const quadrants = [
      [-105, 40, -127, 40, -127, 49, -105, 49, -105, 40],
      [-105, 40, -105, 49, -63, 49, -63, 40, -105, 40],
      [-105, 40, -63, 40, -63, 25, -105, 25, -105, 40],
      [-105, 40, -105, 25, -127, 25, -127, 40, -105, 40],
      [-127, 25, -127, 49, -63, 49, -63, 25, -127, 25],
      [-1000, -1000, -1000, 1000, 1000, 1000, 1000, -1000, -1000, -1000],
    ];
    const SET_POLYGONS_URL     = '/quadtree/set_polygons';
    const SET_POLYGONS_OPTIONS = {
      method: 'POST',
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Content-Type': 'application/json',
        'Access-Control-Allow-Headers': 'Content-Type'
      },
      body: JSON.stringify(
        {name: 'test_quads', polygon_offset: [0, 1], ring_offset: [0, 5], points: quadrants[which]})
    };
    console.log(SET_POLYGONS_OPTIONS.body);
    const result     = await fetch(SERVER + ':' + PORT + SET_POLYGONS_URL, SET_POLYGONS_OPTIONS)
    const resultJson = await result.json();
    return resultJson.params.name;
  };

  const fetchQuadrants = async (csvName, polygonName, props) => {
    /*
      fetchQuadrants is used to fetch the points along each of the four test
      quadrants. This is used to test the performance of the quadtree.
      */
    const FETCH_QUADRANTS_URL     = '/quadtree/get_points/' + csvName + '/' + polygonName;
    const FETCH_QUADRANTS_OPTIONS = {
      method: 'GET',
      headers: {'access-control-allow-origin': '*'},
    };
    const remotePoints =
      await fetch(SERVER + ':' + PORT + FETCH_QUADRANTS_URL, FETCH_QUADRANTS_OPTIONS)
    const arrowTable = await tableFromIPC(remotePoints);
    const hostPoints = arrowTable.getChildAt(0).toArray();
    return hostPoints;
  };

  const fetchPoints =
    async (csvName, props) => {
    const FETCH_POINTS_URL     = '/particles/get_shader_column';
    const FETCH_POINTS_OPTIONS = {
      method: 'GET',
      headers: {'access-control-allow-origin': '*', 'access-control-allow-headers': 'Content-Type'},
    };
    /*
     fetchPoints uses props.currentWorldCoords to request new display points as
     the viewport is changed.

     This is not important now because: On my machine there are ample resources
     to display all 67m points without viewport culling.

     However, the goal of the app is to provide realtime point budget updates, so
     this needs to be factored into an effective module for viewport culling, as well
     as realtime point streaming based on framerate limitations.
     */
    /*
     Do nothing if fetching is already happening
     */
    if (props.fetching === true) return;
    /*
     Fetch path either fetches all points, or takes four additional path parameters
     for xmin, xmas, ymin, and ymax.
     */
    let fetch_path = SERVER + ':' + PORT + FETCH_POINTS_URL + '/' + csvName;
    if (props.currentWorldCoords.xmin !== undefined &&
        props.currentWorldCoords.xmax !== undefined &&
        props.currentWorldCoords.ymin !== undefined &&
        props.currentWorldCoords.ymax !== undefined) {
      const path_tail = '/' + props.currentWorldCoords.xmin + '/' + props.currentWorldCoords.xmax +
                        '/' + props.currentWorldCoords.ymin + '/' + props.currentWorldCoords.ymax;
      fetch_path = fetch_path + path_tail;
    }
    props.fetching = true;
    console.log('fetching');
    const remotePoints = await fetch(fetch_path, FETCH_POINTS_OPTIONS);
    console.log('fetched');
    props.fetching = false;
    /*
     if remotePoints.ok is false, something went wrong in the fetch. Don't try
     to serialize the points from arrow, just print the error message.
     */
    console.log(remotePoints.ok);
    if (remotePoints.ok) {
      const arrowTable = await tableFromIPC(remotePoints);
      const hostPoints = arrowTable.getChild('gpu_buffer').toArray();
      console.log('Fetched ' + hostPoints.length / 2 + ' points.');
      /*
       Render the points
       */
      // Try rendering in batches
      function sleep(ms) { return new Promise(resolve => setTimeout(resolve, ms)); }

      console.log(hostPoints.length);
      /* Loop over the same batches of points endlessly */
      const points = await drawParticles({hostPoints: hostPoints, props});
      points(props);
      /*
      const batchSize = 1000000;
      let i = 0;
      while (true) {
        i           = (i + batchSize) % hostPoints.length;
        const batch = hostPoints.slice(i, i + batchSize);
        await sleep(1000);
        drawParticles({hostPoints: batch, props});
      }
      */
    } else {
      console.log('Unable to fetch');
      console.log(remotePoints);
    }
  }

  const fetchPointsWithEngine = async (csvName, engine, props) => {
    const FETCH_POINTS_URL     = '/particles/get_shader_column';
    const FETCH_POINTS_OPTIONS = {
      method: 'GET',
      headers: {'access-control-allow-origin': '*', 'access-control-allow-headers': 'Content-Type'},
    };
    let fetch_path     = SERVER + ':' + PORT + FETCH_POINTS_URL + '/' + csvName;
    const remotePoints = await fetch(fetch_path, FETCH_POINTS_OPTIONS);
    if (remotePoints.ok) {
      const arrowTable = await tableFromIPC(remotePoints);
      const hostPoints = arrowTable.getChild('gpu_buffer').toArray();
      console.log('Fetched ' + hostPoints.length / 2 + ' points.');
      function sleep(ms) { return new Promise(resolve => setTimeout(resolve, ms)); }
      console.log(engine);
      console.log(hostPoints);
      await engine.subdata(hostPoints);
    } else {
      console.log('Unable to fetch');
      console.log(remotePoints);
    }
  };

  const fetchQuadtree =
    async (csvName, props) => {
    const quadtreeName = await createQuadtree(csvName, props);
    for (let i = 0; i < 400; i++) {
      const which       = i % 6;
      const polygonName = await setPolygon(which, props);
      const hostPoints  = await fetchQuadrants(quadtreeName, polygonName, props);
      console.log(hostPoints.length);
      drawParticles({hostPoints: hostPoints, props});
      function sleep(ms) { return new Promise(resolve => setTimeout(resolve, ms)); }
      await sleep(1000);
    }
  }

  const fetchQuadtreeWithEngine =
    async (csvName, engine, props) => {
    const quadtreeName = await createQuadtree(csvName, props);
    for (let i = 0; i < 400; i++) {
      const which             = i % 6;
      const polygonName       = await setPolygon(which, props);
      const hostPoints        = await fetchQuadrants(quadtreeName, polygonName, props);
      props.displayPointCount = hostPoints.length;
      engine.subdata(hostPoints);
      function sleep(ms) { return new Promise(resolve => setTimeout(resolve, ms)); }
      await sleep(1000);
    }
  }

  /*
   Send the initial request to load the csv file on the server.

   Then render the points that were loaded on the server, then render the background.
   */
  var csvName = undefined;
  try {
    const readCsvResultPromise = await fetch(SERVER + ':' + PORT + READ_CSV_URL, READ_CSV_OPTIONS);
    const readCsvResult        = await readCsvResultPromise.json()
    csvName                    = readCsvResult.params.filename;
    background(props);
    const engine = await particlesEngine(props);
    // fetchPointsWithEngine(csvName, engine, props);
    fetchQuadtreeWithEngine(csvName, engine, props);
    // fetchQuadtree(csvName, props);
    // fetchPoints(csvName, props);
  } catch (e) { console.log(e); }
})();

/*
 Placeholder for working React functionality. The app currently bypasses all React.
 */
ReactDOM.render(React.createElement(App), document.getElementById('root'));
