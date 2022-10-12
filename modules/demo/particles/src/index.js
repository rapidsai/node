/**
 * Copyright (c) 2022 NVIDIA Corporation
 */

// import drawCube from "./drawBackground"

import React from 'react';
import ReactDOM from 'react-dom';

import App from './App';
import background from './background';
import points from './points';

const regl = require('regl')();
const mat4 = require('gl-mat4');

const props = {
  zoomLevel: 0,
  centerX: 0,
  centerY: 0,
  angle: 0
};

window.addEventListener('wheel', (event) => {
  const zoom      = event.deltaY > 0 ? 1 : -1;
  props.zoomLevel = props.zoomLevel + zoom;
});

window.addEventListener('mousedown', (event) => { props.isHeld = true; });
window.addEventListener('mouseup', (event) => { props.isHeld = false; });
window.addEventListener('mousemove', (event) => {
  if (props.isHeld) {
    props.centerX = props.centerX + event.movementX;
    props.centerY = props.centerY + event.movementY;
  }
});

setInterval(() => {props.angle = (props.angle + 1)}, 16);

const SERVER           = 'http://localhost';
const PORT             = '3010';
const READ_CSV_URL     = '/gpu/DataFrame/ReadCSV';
const READ_CSV_OPTIONS = {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: 'points.csv'
};
const FETCH_POINTS_URL = '/particles/get_shader_column';

(async () => {
  try {
    const readCsvResult = await fetch(SERVER + ':' + PORT + READ_CSV_URL);
    const remotePoints  = await fetch(SERVER + ':' + PORT + FETCH_POINTS_URL);
  } finally {
    points(props);
    background(props);
  }
})();

ReactDOM.render(React.createElement(App), document.getElementById('root'));
