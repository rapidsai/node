/**
 * Copyright (c) 2022 NVIDIA Corporation
 */

// import drawCube from "./drawBackground"

const regl = require('regl')();
const mat4 = require('gl-mat4');

const NUM_POINTS = 8
const VERT_SIZE = 4 * (4 + 3)

const pointBuffer = regl.buffer([
  0, 0, 0, 1,
  1.0, 0, 0,
  1, 0, 0, 1,
  0, 1.0, 0,
  0, 1, 0, 1,
  0, 0, 1.0,
  1, -1, 0, 1,
  1.0, 1.0, 0,
  -1, 1, 0, 1,
  1.0, 0, 1.0,
  1, 1, 0, 1,
  0, 1.0, 1.0,
  0, -1, 0, 1,
  1.0, 1.0, 1.0,
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
        [props.centerX / 100, props.centerY / 100, lookAtZ],
        [props.centerX / 100, props.centerY / 100, 0],
        [0, 1, 0]);
      const translation = mat4.translate([], result, [1, 1, 0]);
      const rotation = mat4.rotate([], translation, t, [0, 0, 1]);
      return rotation;
    },
    scale: ({tick}, props) => {
      return 50 - (25 + props.zoomLevel);
    },
    projection: ({ viewportWidth, viewportHeight }) =>
      mat4.frustum([],
        -500, 500, 300, -300, -1000, 1),
    time: ({ tick }) => tick * 0.001
  },

  count: NUM_POINTS,

  primitive: 'points'
})

const props = {
  zoomLevel: 0,
  centerX: 0,
  centerY: 0,
  angle: 0
};

window.addEventListener('wheel', (event) => {
  const zoom = event.deltaY > 0 ? 1 : -1;
  props.zoomLevel = props.zoomLevel + zoom;
});

window.addEventListener('mousedown', (event) => {
  props.isHeld = true;
});
window.addEventListener('mouseup', (event) => {
  props.isHeld = false;
});
window.addEventListener('mousemove', (event) => {
  if(props.isHeld) {
    props.centerX = props.centerX + event.movementX;
    props.centerY = props.centerY + event.movementY;
  }
});

setInterval(() => {
  props.angle = (props.angle + 1)
}, 16);

var cubePosition = [
  [-0.5, +0.5, +0.5], [+0.5, +0.5, +0.5], [+0.5, -0.5, +0.5], [-0.5, -0.5, +0.5], // positive z face.
  [+0.5, +0.5, +0.5], [+0.5, +0.5, -0.5], [+0.5, -0.5, -0.5], [+0.5, -0.5, +0.5], // positive x face
  [+0.5, +0.5, -0.5], [-0.5, +0.5, -0.5], [-0.5, -0.5, -0.5], [+0.5, -0.5, -0.5], // negative z face
  [-0.5, +0.5, -0.5], [-0.5, +0.5, +0.5], [-0.5, -0.5, +0.5], [-0.5, -0.5, -0.5], // negative x face.
  [-0.5, +0.5, -0.5], [+0.5, +0.5, -0.5], [+0.5, +0.5, +0.5], [-0.5, +0.5, +0.5], // top face
  [-0.5, -0.5, -0.5], [+0.5, -0.5, -0.5], [+0.5, -0.5, +0.5], [-0.5, -0.5, +0.5]  // bottom face
]

var cubeUv = [
  [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], // positive z face.
  [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], // positive x face.
  [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], // negative z face.
  [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], // negative x face.
  [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], // top face
  [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]  // bottom face
]

const cubeElements = [
  [2, 1, 0], [2, 0, 3],       // positive z face.
  [6, 5, 4], [6, 4, 7],       // positive x face.
  [10, 9, 8], [10, 8, 11],    // negative z face.
  [14, 13, 12], [14, 12, 15], // negative x face.
  [18, 17, 16], [18, 16, 19], // top face.
  [20, 21, 22], [23, 20, 22]  // bottom face
]

const drawCube = regl({
  frag: `
  precision mediump float;
  varying vec2 vUv;
  uniform sampler2D tex;
  void main () {
    gl_FragColor = texture2D(tex,vUv);
  }`,
  vert: `
  precision mediump float;
  attribute vec3 position;
  attribute vec2 uv;
  varying vec2 vUv;
  uniform mat4 projection, view;
  void main() {
    vUv = uv;
    gl_Position = projection * view * vec4(position, 1);
  }`,
  attributes: {
    position: cubePosition,
    uv: cubeUv
  },
  elements: cubeElements,
  uniforms: {
    view: ({tick}) => {
      const t = 0.01 * tick
      return mat4.lookAt([],
                         [5 * Math.cos(t), 2.5 * Math.sin(t), 5 * Math.sin(t)],
                         [0, 0.0, 0],
                         [0, 1, 0])
    },
    projection: ({viewportWidth, viewportHeight}) =>
      mat4.perspective([],
                       Math.PI / 4,
                       viewportWidth / viewportHeight,
                       0.01,
                       10),
    tex: regl.prop('data')
  }
})

const data = regl.texture({
  width: 2,
  height: 2,
  data: [
    255, 255, 255, 255, 0, 0, 0, 0,
    255, 0, 255, 255, 0, 0, 255, 255
  ]
})
/*
console.log(process.env.PUBLIC_URL);
regl.frame(() => {
  regl.clear({
    color: [0, 0, 0, 255],
    depth: 1
  });
  drawCube({data});
});
*/

const tick = regl.frame(() => {
  regl.clear({
    depth: 1,
    color: [0, 0, 0, 0]
  })
  drawParticles(props);
  drawCube({data})
});
