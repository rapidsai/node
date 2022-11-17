/**
 * Copyright (c) 2022 NVIDIA Corporation
 */

const regl                                             = require('regl')();
const mat4                                             = require('gl-mat4');
const {getPointsViewMatrix, getPointsProjectionMatrix} = require('./matrices');

const NUM_POINTS = 400;
const VERT_SIZE  = 4 * (2)

const numGenerator = {
  * [Symbol.iterator]() {
      let i = 0;
      while (i < NUM_POINTS * 4) {
        yield [Math.random() * 200 - 20.0,
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

let generatedHostPoints              = [...numGenerator];
export default ({hostPoints, props}) => {
  hostPoints          = hostPoints;
  const pointBuffer   = regl.buffer(hostPoints);
  const drawParticles = regl({
    vert: `
precision mediump float;
attribute vec2 pos;
uniform float scale;
uniform float time;
uniform mat4 view, projection;
varying vec3 fragColor;
void main() {
  vec2 position = pos.xy;
  gl_PointSize = scale;
  gl_Position = projection * view * vec4(position, 1, 1);
  fragColor = vec3(0, 0, 0);
}`,
    frag: `
precision lowp float;
varying vec3 fragColor;
void main() {
  if (length(gl_PointCoord.xy - 0.5) > 0.5) {
    discard;
  }
  gl_FragColor = vec4(fragColor, 0.5);
}`,
    attributes: {
      pos: {buffer: pointBuffer, stride: VERT_SIZE, offset: 0},
    },
    uniforms: {
      view: ({tick}, props)  => getPointsViewMatrix(props),
      scale: ({tick}, props) => { return Math.max(1.5, -props.zoomLevel); },
      projection: ({viewportWidth, viewportHeight}) => getPointsProjectionMatrix(props),
      time: ({tick})                                => tick * 0.001
    },
    count: hostPoints.length / VERT_SIZE,
    primitive: 'points'
  })

  const tick = regl.frame(() => {
    regl.clear({depth: 1, color: [0, 0, 0, 0]});
    drawParticles(props);
  });
}
