/**
 * Copyright (c) 2022 NVIDIA Corporation
 */

const regl                                             = require('regl')();
const mat4                                             = require('gl-mat4');
const {getPointsViewMatrix, getPointsProjectionMatrix} = require('./matrices');

/*
 A random points generator for testing.
 */
const NUM_POINTS   = 400;
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
let generatedHostPoints = [...numGenerator];

/*
 The points function that renders points into the same world
 view as the background image.
 */
/*
 A constant that defines the stride of the input buffer for rendering.
 */
export const drawParticles =
  ({hostPoints, props}) => {
    const buffer = regl.buffer({usage: 'dynamic', length: hostPoints.length * 4});
    buffer.subdata(hostPoints);
    const re = regl({
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
        pos: {buffer: regl.buffer(hostPoints), stride: 8, offset: 0},
      },
      uniforms: {
        view: ({tick}, props)  => getPointsViewMatrix(props),
        scale: ({tick}, props) => { return Math.max(1.5, -props.zoomLevel); },
        projection: ({viewportWidth, viewportHeight}) => getPointsProjectionMatrix(props),
        time: ({tick})                                => tick * 0.001
      },
      count: hostPoints.length,
      primitive: 'points'
    });
    console.log(hostPoints);
    return re(props);
  }

export const drawBuffer =
  (buffer, props) => {
    const re = regl({
      vert: `
precision mediump float;
attribute vec2 pos;
uniform float scale;
uniform float time;
uniform mat4 view, projection;
varying vec3 fragColor;
void main() {
  vec2 position = pos.xy;
  gl_PointSize = scale * 10.0;
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
        pos: {buffer: buffer, stride: 8, offset: 0},
      },
      uniforms: {
        view: ({tick}, props)  => getPointsViewMatrix(props),
        scale: ({tick}, props) => { return Math.max(1.5, -props.zoomLevel); },
        projection: ({viewportWidth, viewportHeight}) => getPointsProjectionMatrix(props),
        time: ({tick})                                => tick * 0.001
      },
      count: 4,
      primitive: 'points'
    });
    return re;
  }

export const particlesEngine = (props) => {
  const buffer = regl.buffer({usage: 'dynamic', length: props.pointBudget * 8});
  let assigned = 0;
  buffer.subdata([-105, 40, -106, 40, -106, 41, -105, 41], 0);
  const particles = drawBuffer(buffer, props);
  const tick      = regl.frame(() => {
    regl.clear({depth: 1, color: [0, 0, 0, 0]});
    particles(props);
  });

  const subdata = (hostPoints, range) => {
    buffer.subdata(hostPoints, range);
    assigned = assigned + hostPoints.length;
  };

  return {subdata: subdata};
}
