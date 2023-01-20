/**
 * Copyright (c) 2022 NVIDIA Corporation
 */

const regl                                             = require('regl')();
const {getPointsViewMatrix, getPointsProjectionMatrix} = require('./matrices');

/*
 The points function that renders points into the same world
 view as the background image.
 */
const drawBuffer =
  async (buffer, props) => {
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
      pos: {buffer: buffer, stride: 8, offset: 0},
    },
    uniforms: {
      view: ({tick}, props)  => getPointsViewMatrix(props),
      scale: ({tick}, props) => { return Math.max(1.5, Math.pow(props.zoomLevel, 1 / 2.6)); },
      projection: ({viewportWidth, viewportHeight}) => getPointsProjectionMatrix(props),
      time: ({tick})                                => tick * 0.001
    },
    count: props.pointBudget,
    primitive: 'points'
  });
  return re;
}

export const particlesEngine = async (props) => {
  const buffer = regl.buffer({usage: 'dynamic', type: 'float', length: props.pointBudget * 4});
  const tick   = regl.frame(async () => {
    regl.clear({depth: 1, color: [0, 0, 0, 0]});
    const particles = await drawBuffer(buffer, props);
    particles(props);
  });

  const subdata = async (hostPoints, props) => {
    // buffer(hostPoints);
    console.log(props.pointOffset);
    console.log(hostPoints.length);
    buffer.subdata(hostPoints, props.pointOffset * 4);
  };

  return {subdata: subdata};
}
