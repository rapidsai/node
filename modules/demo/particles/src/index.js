/**
 * Copyright (c) 2022 NVIDIA Corporation
 */

// import drawCube from "./drawBackground"

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
    props.centerX                 = props.centerX + event.movementX;
    props.centerY                 = props.centerY + event.movementY;
    props.screenWidth             = document.documentElement.clientWidth;
    props.screenHeight            = document.documentElement.clientHeight;
    const zoom                    = event.deltaY > 0 ? -1 : 1;
    props.zoomLevel               = props.zoomLevel + zoom;
    const newWorldCoords          = getScreenToWorldCoords(props);
    props.currentWorldCoords.xmin = newWorldCoords[0];
    props.currentWorldCoords.xmax = newWorldCoords[8];
    props.currentWorldCoords.ymin = newWorldCoords[9];
    props.currentWorldCoords.ymax = newWorldCoords[1];
    console.log(props);
    // fetchPoints(csvPath, props);
  });
  window.addEventListener('mousedown', (event) => { props.isHeld = true; });
  window.addEventListener('mouseup', (event) => { props.isHeld = false; });
  window.addEventListener('mousemove', (event) => {
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
   Deprecated interval rotated points when I was working on a better
   coordinate system understanding of them.
   */
  setInterval(() => {props.angle = (props.angle + 1)}, 16);

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
    // body: '"NAD_30m.csv"',
    body: '"NAD_State_ZIP_LonLat.csv"',
    // body: '"shuffled.csv"',
    // body: '"NAD_Shuffled_100000.csv"',
  };
  const FETCH_POINTS_URL     = '/particles/get_shader_column';
  const FETCH_POINTS_OPTIONS = {
    method: 'GET',
    headers: {'access-control-allow-origin': '*', 'access-control-allow-headers': 'Content-Type'},
  };

  const fetchPoints =
    async (csvPath, props) => {
    if (props.fetching === true) return;
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
    console.log(remotePoints.ok);
    if (remotePoints.ok) {
      const arrowTable = await tableFromIPC(remotePoints);
      console.log('made table');
      const hostPoints = arrowTable.getChild('gpu_buffer').toArray();
      console.log('got points');
      console.log('Fetched ' + hostPoints.length / 7 + ' points.');
      points({hostPoints, props});
    } else {
      console.log('Unable to fetch');
      console.log(remotePoints);
    }
  }

  var csvPath = undefined;

  var hostPoints   = undefined;
  var renderPoints = undefined;
  try {
    const readCsvResultPromise = await fetch(SERVER + ':' + PORT + READ_CSV_URL, READ_CSV_OPTIONS);
    const readCsvResult        = await readCsvResultPromise.json()
    csvPath                    = readCsvResult.params;
    fetchPoints(csvPath, props);
    background(props);
  } catch (e) { console.log(e); }
})();

ReactDOM.render(React.createElement(App), document.getElementById('root'));
