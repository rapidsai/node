/**
 * Copyright (c) 2022 NVIDIA Corporation
 */

const regl = require('regl')();
const mat4 = require('gl-mat4');

const getBackgroundViewMatrix = (props) => {
  const t       = 0;  // 0.015 * (props.angle);
  const lookAtZ = 4 * Math.pow(1.2, props.zoomLevel);
  const result =
    mat4.lookAt([],
                [-props.centerX / lookAtZ / 10, -props.centerY / lookAtZ / 10, lookAtZ],
                [-props.centerX / lookAtZ / 10, -props.centerY / lookAtZ / 10, 0],
                [0, 1, 0]);
  const translation = mat4.translate([], result, [0, 0, 0]);
  const rotation    = mat4.rotate([], translation, t, [t, t, 1]);
  return rotation;
};

export default (props) => {
  var cubePosition = [
    //[-0.5, +0.5, 0.1], [+0.5, +0.5, 0.1], [+0.5, -0.5, 0.1], [-0.5, -0.5, 0.1] // positive z
    // face.
    [-50, 50, 0.1],
    [50, 50, 0.1],
    [50, -50, 0.1],
    [-50, -50, 0.1]
  ];

  var cubeUv = [
    [1.0, 0.0],
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],  // positive z face.
  ];

  const cubeElements = [
    [0, 1, 2],
    [0, 3, 2]  // positive z face.
  ];
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
  vec3 pos = -position;
  gl_Position = projection * view * vec4(pos, 1);
}`,
    attributes: {position: cubePosition, uv: cubeUv},
    elements: cubeElements,
    uniforms: {
      projection: ({viewportWidth, viewportHeight}) => props.projectionMatrix,
      view: ({tick}, props)                         => { return props.props.backgroundViewMatrix; },
      tex: regl.prop('data')
    }
  })

  // Try to put an image in a context2d buffer
  const backgroundCanvas = document.createElement('div');
  backgroundCanvas.innerHTML =
    '<canvas id=\'hiddenBackground\' width=3000 height=3000 style=\'visibility: hidden\'/>'
  const image = new Image();
  image.src   = './usa_map.png';
  image.style = 'visibility: hidden';
  backgroundCanvas.appendChild(image);
  document.body.appendChild(backgroundCanvas);
  const canvas = document.getElementById('hiddenBackground');
  image.onload = () => {
    const context = canvas.getContext('2d');
    context.drawImage(image, 0, 0);
    const imWidth   = image.width;
    const imHeight  = image.height;
    const imageData = context.getImageData(0, 0, imWidth, imHeight);

    const data = regl.texture({width: imWidth, height: imHeight, data: imageData.data});
    const tick = regl.frame(() => {
      regl.clear({depth: 1, color: [0, 0, 0, 0]});
      props.backgroundViewMatrix = getBackgroundViewMatrix(props);
      drawCube({data, props});
    });
  }
}
