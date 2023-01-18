/**
 * Copyright (c) 2022 NVIDIA Corporation
 */

import React from 'react';
import ReactDOM from 'react-dom';

import App from './App';
import background from './background';
const {drawParticles, particlesEngine}                         = require('./points');
const {getQuadtreePoints, setPolygon, readCsv, createQuadtree} = require('./requests');
const {computeTiming}                                          = require('./computeTiming');

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
    pointBudget: 60000000,
    pointOffset: 0
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

  function separateCircle(radius) {
    const polygons = [];
    const centerX  = -105;
    const centerY  = 39.73;
    for (let i = 0; i < 144; i++) {
      const angle1 = (i * 2.5) / 45 * Math.PI;
      const angle2 = (i * 2.5 + 2.5) / 45 * Math.PI;
      const points = [
        centerX,
        centerY,
        radius * Math.cos(angle1) + centerX,
        radius * Math.sin(angle1) + centerY,
        radius * Math.cos(angle2) + centerX,
        radius * Math.sin(angle2) + centerY,
        centerX,
        centerY
      ];
      polygons.push(points);
    }
    return polygons;
  }

  /*const quadrants = [
    [-105, 40, -127, 40, -127, 49, -105, 49, -105, 40],
    [-105, 40, -105, 49, -63, 49, -63, 40, -105, 40],
    [-105, 40, -63, 40, -63, 25, -105, 25, -105, 40],
    [-105, 40, -105, 25, -127, 25, -127, 40, -105, 40],
    [-127, 25, -127, 49, -63, 49, -63, 25, -127, 25],
    [-1000, -1000, -1000, 1000, 1000, 1000, 1000, -1000, -1000, -1000],
  ];
  const circleSlices         = separateCircle(40);
  */

  const fetchPoints = async (csvName, engine, props) => {
    const hostPoints = await fetchPoints(csvName);
    engine.subdata(hostPoints, props);
  };

  const fetchQuadtree = async (csvName, engine, props) => {
    const quadtreeName = await createQuadtree(csvName, {x: 'Longitude', y: 'Latitude'});
    let i              = 0;
    const polygons     = separateCircle(40);
    while (props.pointOffset < props.pointBudget) {
      const which = i % 36;
      i++;
      const polygonName = await setPolygon('p1', polygons[which]);
      const hostPoints  = await getQuadtreePoints(quadtreeName, polygonName, 1500000);
      engine.subdata(hostPoints, props);
      props.pointOffset = (props.pointOffset % props.pointBudget) + hostPoints.length;
      const sleep =
        (milliseconds) => { return new Promise(resolve => setTimeout(resolve, milliseconds)) };
      // await sleep(1000);
    }
  };
  /*
   Send the initial request to load the csv file on the server.

   Then render the points that were loaded on the server, then render the background.
   */
  var csvName = undefined;
  try {
    background(props);
    const csvName = await readCsv('shuffled.csv');
    const engine  = await particlesEngine(props);
    // fetchPoints(csvName, engine, props);
    computeTiming();
    fetchQuadtree(csvName, engine, props);
  } catch (e) { console.log(e); }
})();

/*
 Placeholder for working React functionality. The app currently bypasses all React.
 */
ReactDOM.render(React.createElement(App), document.getElementById('root'));
