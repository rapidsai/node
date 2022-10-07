/**
 * Copyright (c) Mik BRY
 * mik@mikbry.com
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React, { useEffect } from 'react';

const mat4 = require('gl-mat4')

const NUM_POINTS = 8
const VERT_SIZE = 4 * (4 + 3)


const ParticlesView = (props) => {
  useEffect(() => {
    const canvas = document.getElementById('reglCanvas');
    canvas.height = 1000;
    canvas.width = 1000;
    const regl = require('regl')(canvas.getContext('webgl'));
    const pointBuffer = regl.buffer([
      0, 0, 0, 1,
      1.0, 0, 0,
      //
      1, 0, 0, 1,
      0, 1.0, 0,
      //
      0, 1, 0, 1,
      0, 0, 1.0,
      //
      1, -1, 0, 1,
      1.0, 1.0, 0,
      //
      -1, 1, 0, 1,
      1.0, 0, 1.0,
      //
      1, 1, 0, 1,
      0, 1.0, 1.0,
      //
      0, -1, 0, 1,
      1.0, 1.0, 1.0,
      //
      -1, -1, 0, 1,
      0, 0, 0,
    ]);

    const drawParticles = regl({
      vert: `
    precision mediump float;
    attribute vec4 freq;
    attribute vec3 color;
    uniform float scale;
    uniform float time;
    uniform mat4 view, projection;
    varying vec3 fragColor;
    void main() {
      vec3 position = freq.xyz; //cos(freq.xyz * time + phase.xyz);
      gl_PointSize = scale;
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
        color: {
          buffer: pointBuffer,
          stride: VERT_SIZE,
          offset: 16
        }
      },

      uniforms: {
        view: ({ tick }, props) => {
          const t = 0.005 * (props.angle);
          const lookAtZ = 4 * Math.pow(1.1, props.zoomLevel);
          const result = mat4.lookAt([],
            [0, 0, lookAtZ],
            [0, 0, 0],
            [0, 1, 0]);
          const translation = mat4.translate([], result, [1, 1, 0]);
          const rotation = mat4.rotate([], translation, t, [0, 0, 1]);
          return rotation;
        },
        scale: () => {
          return 50 - (25 + props.zoomLevel);
        },
        projection: ({ viewportWidth, viewportHeight }) =>
          /*
            mat4.perspective([],
              Math.PI / 4,
              viewportWidth / viewportHeight,
              0.01,
              1000),
              */
          mat4.frustum([],
            -500, 500, -500, 500, -1000, 1),
        time: ({ tick }) => tick * 0.001
      },

      count: NUM_POINTS,

      primitive: 'points'
    })


    const tick = regl.frame(() => {
      regl.clear({
        depth: 1,
        color: [0, 0, 0, 0]
      })
      drawParticles(props);
    });
    return () => {
      regl.destroy();
    };
  })
  return <canvas id="reglCanvas" />;
}

export default ParticlesView;
