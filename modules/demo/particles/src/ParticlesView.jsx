/**
 * Copyright (c) Mik BRY
 * mik@mikbry.com
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React, { useEffect, useRef } from 'react';
import PropTypes from 'prop-types';
import WebGL from './WebGL';

const regl = require('regl')()
const mat4 = require('gl-mat4')

const NUM_POINTS = 8
const VERT_SIZE = 4 * (4 + 1 + 3)

const pointBuffer = regl.buffer([
  0, 0, 0, 1,
  1.0,
  1.0, 0, 0,
  //
  1, 0, 0, 1,
  1,
  0, 1.0, 0,
  //
  0, 1, 0, 1,
  1,
  0, 0, 1.0,
  //
  1, -1, 0, 1,
  1,
  1.0, 1.0, 0,
  //
  -1, 1, 0, 1,
  1,
  1.0, 0, 1.0,
  //
  1, 1, 0, 1,
  1,
  0, 1.0, 1.0,
  //
  0, -1, 0, 1,
  1,
  1.0, 1.0, 1.0,
  //
  -1, -1, 0, 1,
  1,
  0, 0, 0,
]);  /*Array(NUM_POINTS).fill().map(function () {
  const color = [Math.random() * 255, Math.random() * 255, Math.random(0) * 255, 255]; //  hsv2rgb(Math.random() * 360, 0.6, 1)
  return [
    // freq
    Math.random() * 10,
    Math.random() * 10,
    Math.random() * 10,
    Math.random() * 10,
    // phase
    1, //2.0 * Math.PI * Math.random(),
    1, //2.0 * Math.PI * Math.random(),
    1, //2.0 * Math.PI * Math.random(),
    2.0 * Math.PI * Math.random(),
    // color
    color[0] / 255, color[1] / 255, color[2] / 255
  ]
}))*/

const drawParticles = regl({
  vert: `
    precision mediump float;
    attribute vec4 freq;
    attribute float scale;
    attribute vec3 color;
    uniform float time;
    uniform mat4 view, projection;
    varying vec3 fragColor;
    void main() {
      vec3 position = freq.xyz * (scale * 0.5); //cos(freq.xyz * time + phase.xyz);
      gl_PointSize = 25.0; //* (1.0 + cos(freq.w * time + phase.w));
      gl_Position = projection * view * vec4(position, 1);
      fragColor = color;
    }`,

  frag: `
    precision lowp float;
    varying vec3 fragColor;
    void main() {
      if (length(gl_PointCoord.xy - 0.5) > 0.5) {
        discard;
      }
      gl_FragColor = vec4(fragColor, 1);
    }`,

  attributes: {
    freq: {
      buffer: pointBuffer,
      stride: VERT_SIZE,
      offset: 0
    },
    scale: {
      buffer: pointBuffer,
      stride: VERT_SIZE,
      offset: 16
    },
    color: {
      buffer: pointBuffer,
      stride: VERT_SIZE,
      offset: 20
    }
  },

  uniforms: {
    view: ({ tick }) => {
      const t = 0.01 * tick
      return mat4.lookAt([],
        [0, 0, -10], // * Math.cos(t), 2.5, 30 * Math.sin(t)],
        [0, 0, 0],
        [0, 1, 0])
    },
    projection: ({ viewportWidth, viewportHeight }) =>
      mat4.perspective([],
        Math.PI / 4,
        viewportWidth / viewportHeight,
        0.01,
        1000),
    time: ({ tick }) => tick * 0.001
  },

  count: NUM_POINTS,

  primitive: 'points'
})

const ParticlesView = () => {
  const ref = useRef();

  useEffect(() => {
    regl.frame(() => {
      regl.clear({
        depth: 1,
        color: [0, 0, 0, 0]
      })

      drawParticles()
    })
  })
  return <canvas ref={ref} />
}

export default ParticlesView;
