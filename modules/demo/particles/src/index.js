/**
 * Copyright (c) 2022 NVIDIA Corporation
 */

import React from 'react';
import ReactDOM from 'react-dom';

import App from './App';
import background from './background';
const {drawParticles, particlesEngine} = require('./points');
const {getQuadtreePoints, getQuadtreePointCount, setPolygon, readCsv, createQuadtree, release} =
  require('./requests');
const {computeTiming} = require('./computeTiming');

const {tableFromIPC}                                 = require('apache-arrow');
const mat4                                           = require('gl-mat4');
const {getScreenToWorldCoords, getCurrentOrthoScale} = require('./matrices');

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
    zoomLevel: 2.0,
    angle: 0,
    screenWidth: document.documentElement.clientHeight,
    screenHeight: document.documentElement.clientWidth,
    centerX: document.documentElement.clientWidth / 2.0,
    centerY: document.documentElement.clientHeight / 2.0,
    currentWorldCoords: {xmin: undefined, xmax: undefined, ymin: undefined, ymax: undefined},
    fetching: false,
    pointBudget: 180000000,
    pointsPerRequest: 1000000,
    pointOffset: 0,
    quads: {
      totalPoints: undefined,
      displayedPoints: 0,
      pointOffsets: [],
    },
    done: {},
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
    // props.zoomLevel    = props.zoomLevel + zoom;
    props.zoomLevel = event.deltaY > 0 ? props.zoomLevel / 1.2 : props.zoomLevel * 1.2;
    /*
     newWorldCoords defines the bounding in world-coordinates of the current
     view. This is used to update the points using server based viewport culling.
     */
    const newWorldCoords          = getScreenToWorldCoords(props);
    props.currentWorldCoords.xmin = newWorldCoords[0];
    props.currentWorldCoords.ymin = newWorldCoords[1];
    props.currentWorldCoords.xmax = newWorldCoords[8];
    props.currentWorldCoords.ymax = newWorldCoords[9];
    console.log(props.zoomLevel);
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
      const moveX                   = event.movementX * getCurrentOrthoScale(props) * 0.55;
      const moveY                   = event.movementY * getCurrentOrthoScale(props) * 0.85;
      props.centerX                 = props.centerX + moveX;
      props.centerY                 = props.centerY + moveY;
      const newWorldCoords          = getScreenToWorldCoords(props);
      props.currentWorldCoords.xmin = newWorldCoords[0];
      props.currentWorldCoords.ymin = newWorldCoords[1];
      props.currentWorldCoords.xmax = newWorldCoords[8];
      props.currentWorldCoords.ymax = newWorldCoords[9];
      console.log(props.currentWorldCoords.ymin,
                  props.currentWorldCoords.xmin,
                  props.currentWorldCoords.ymax,
                  props.currentWorldCoords.xmax)
    }
  });

  const quadrants = [
    [-105, 40, -127, 40, -127, 49, -105, 49, -105, 40],
    [-105, 40, -105, 49, -63, 49, -63, 40, -105, 40],
    [-105, 40, -63, 40, -63, 25, -105, 25, -105, 40],
    [-105, 40, -105, 25, -127, 25, -127, 40, -105, 40],
    //[-1000, -1000, -1000, 1000, 1000, 1000, 1000, -1000, -1000, -1000],
  ];

  const estimatedInitialViewport = [-127, 25, -127, 49, -63, 49, -63, 25, -127, 25];
  const quadPair                 = [-127, 25, -63, 49];

  const makeQuadrants =
    (rectangle, list, depth = 3) => {
      const x1 = rectangle[0];
      const y1 = rectangle[1];
      const x2 = rectangle[2];
      const y2 = rectangle[3];
      const mx = (x1 + x2) / 2;
      const my = (y1 + y2) / 2;
      if (depth < 1) {
        list.unshift([x1, y1, x2, y1, x2, y2, x1, y2, x1, y1]);
        return;
      }
      makeQuadrants([x1, y1, mx, my], list, depth - 1);
      makeQuadrants([mx, y1, x2, my], list, depth - 1);
      makeQuadrants([x1, my, mx, y2], list, depth - 1);
      makeQuadrants([mx, my, x2, y2], list, depth - 1);
    }

  function shuffleArray(array) {
    for (let i = array.length - 1; i > 0; i--) {
      const j              = Math.floor(Math.random() * (i + 1));
      [array[i], array[j]] = [array[j], array[i]];
    }
  };
  async function getPolygonSizes(quadtreeName, polygons, props) {
    const sizes = {};
    for (let i = 0; i < polygons.length; i++) {
      const polygonName        = await setPolygon(polygons[i], polygons[i]);
      const res                = await getQuadtreePointCount(quadtreeName, polygonName);
      const size               = res.count;
      props.quads[polygons[i]] = {totalPoints: size, pointOffsets: [], loadedPoints: 0};
    }
  }
  const fetchQuadtree = async (csvName, engine, props) => {
    const quadtreeName = await createQuadtree(csvName, {x: 'Longitude', y: 'Latitude'});
    let polygons       = [];
    makeQuadrants(quadPair, polygons, 3);
    shuffleArray(polygons);
    await getPolygonSizes(quadtreeName, polygons, props);
    let which = 0;
    while (props.pointOffset < props.pointBudget && polygons.length > 0) {
      console.log(props.quads[polygons[which]]);
      if (props.quads[polygons[which]].totalPoints <= props.quads[polygons[which]].loadedPoints) {
        props.done[polygons[which]] = props.quads[polygons[which]];
        delete props.quads[polygons[which]];
        polygons.splice(which, 1);
        which = which % polygons.length;
        continue;
      }
      const hostPoints =
        await getQuadtreePoints(quadtreeName, polygons[which], props.pointsPerRequest);
      engine.subdata(hostPoints, props);
      const newOffset = (props.pointOffset % props.pointBudget) + hostPoints.length;
      if (props.quads[polygons[which]].loadedPoints === undefined) {
        props.quads[polygons[which]].loadedPoints = 0;
      }
      props.quads[polygons[which]].loadedPoints += hostPoints.length / 2
      props.quads[polygons[which]].pointOffsets.push([props.pointOffset, newOffset]);
      props.pointOffset = newOffset;
      const sleep =
        (milliseconds) => { return new Promise(resolve => setTimeout(resolve, milliseconds)) };
      // await sleep(1000);
      which = (which + 1) % polygons.length;
    }
  };
  /*
   Send the initial request to load the csv file on the server.

   Then render the points that were loaded on the server, then render the background.
   */
  var csvName = undefined;
  try {
    await release();
    background(props);
    const csvName = await readCsv('NAD_r11.txt');
    const engine  = await particlesEngine(props);
    fetchQuadtree(csvName, engine, props);
  } catch (e) { console.log(e); }
})();

/*
 Placeholder for working React functionality. The app currently bypasses all React.
 */
ReactDOM.render(React.createElement(App), document.getElementById('root'));
