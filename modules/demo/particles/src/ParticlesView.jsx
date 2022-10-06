/**
 * Copyright (c) Mik BRY
 * mik@mikbry.com
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React, { useEffect, useRef } from 'react';

const mat4 = require('gl-mat4')

const NUM_POINTS = 8
const VERT_SIZE = 4 * (4 + 1 + 3)


const ParticlesView = (props) => {
  const ref = useRef();

  useEffect(() => {
    const canvas = document.getElementById('reglCanvas');
    canvas.height = 1000;
    canvas.width = 1000;
    const regl = require('regl')(canvas.getContext('webgl'));
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
    ]);

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
        view: ({ tick }, props) => {
          const t = 0.01 * (props.angle + tick);
          const lookAtZ = 2 * Math.pow(1.1, props.zoomLevel);
          if (tick % 100 == 0) {
            console.log(tick);
            console.log(props.zoomLevel);
            console.log(lookAtZ);
          }
          const result = mat4.lookAt([],
            [0, 0, lookAtZ],
            [0, 0, 0],
            [0, 1, 0])
          return mat4.rotate([], result, Math.cos(t), [0, 0, 1]);
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


    const tick = regl.frame(() => {
      regl.clear({
        depth: 1,
        color: [0, 0, 0, 0]
      })
      console.log(props);
      drawParticles(props);
    });
    return () => {
      regl.destroy();
    };
  })
  return <canvas ref={ref} />;
}

export default ParticlesView;
