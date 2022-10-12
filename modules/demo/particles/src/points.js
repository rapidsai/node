/**
 * Copyright (c) 2022 NVIDIA Corporation
 */

const regl                                       = require('regl')();
const mat4                                       = require('gl-mat4');
const {getPointsViewMatrix, getProjectionMatrix} = require('./matrices');

const NUM_POINTS = 10000;
const VERT_SIZE  = 4 * (4 + 3)

const numGenerator = {
  * [Symbol.iterator]() {
      let i = 0;
      while (i < NUM_POINTS * 4) {
        yield [Math.random() * 100 - 20.0,
               Math.random() * 100 - 20.0,
               1.0,
               1.0,
               Math.random() > 0.5 ? 255 : 0,
               Math.random() > 0.5 ? 255 : 0,
               Math.random() > 0.5 ? 255 : 0,
        ]
        i++;
      }
    }
};

const pointBuffer = regl.buffer([...numGenerator]);

export default (props) => {
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
  vec3 position = freq.xyz;
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
      freq: {buffer: pointBuffer, stride: VERT_SIZE, offset: 0},
      color: {buffer: pointBuffer, stride: VERT_SIZE, offset: 16}
    },
    uniforms: {
      view: ({tick}, props)  => getPointsViewMatrix(props),
      scale: ({tick}, props) => { return 40 - (25 + Math.min(props.zoomLevel, 13)); },
      projection: ({viewportWidth, viewportHeight}) => getProjectionMatrix(props),
      time: ({tick})                                => tick * 0.001
    },
    count: NUM_POINTS,
    primitive: 'points'
  })

  const tick = regl.frame(() => {
    regl.clear({depth: 1, color: [0, 0, 0, 0]});
    drawParticles(props);
  });
}
