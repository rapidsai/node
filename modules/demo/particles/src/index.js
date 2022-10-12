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

points(props);
background(props);

ReactDOM.render(React.createElement(App), document.getElementById('root'));
