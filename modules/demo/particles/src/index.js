/**
 * Copyright (c) 2022 NVIDIA Corporation
 */

import React from 'react';
import ReactDOM from 'react-dom';

import App from './App';
import background from './background';
import points from './points';

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
    fetching: false
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
    // fetchPoints(csvPath, props);
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
      // fetchPoints(csvPath, props, newWorldCoords);
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
    body: '"shuffled.csv"',
    // body: '"NAD_Shuffled_100000.csv"',
  };
  const FETCH_POINTS_URL     = '/particles/get_shader_column';
  const FETCH_POINTS_OPTIONS = {
    method: 'GET',
    headers: {'access-control-allow-origin': '*', 'access-control-allow-headers': 'Content-Type'},
  };

  const fetchPoints =
    async (csvPath, props) => {
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
    let fetch_path = SERVER + ':' + PORT + FETCH_POINTS_URL + '/' + csvPath;
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
      points({hostPoints, props});
    } else {
      console.log('Unable to fetch');
      console.log(remotePoints);
    }
  }

  /*
   Send the initial request to load the csv file on the server.

   Then render the points that were loaded on the server, then render the background.
   */
  var csvPath = undefined;
  try {
    const readCsvResultPromise = await fetch(SERVER + ':' + PORT + READ_CSV_URL, READ_CSV_OPTIONS);
    const readCsvResult        = await readCsvResultPromise.json()
    csvPath                    = readCsvResult.params;
    fetchPoints(csvPath, props);
    background(props);
  } catch (e) { console.log(e); }
})();

/*
 Placeholder for working React functionality. The app currently bypasses all React.
 */
ReactDOM.render(React.createElement(App), document.getElementById('root'));
