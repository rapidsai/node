/**
 * Copyright (c) 2022 NVIDIA Corporation
 */

const regl                                                     = require('regl')();
const mat4                                                     = require('gl-mat4');
const {getBackgroundProjectionMatrix, getBackgroundViewMatrix} = require('./matrices');

export default (props) => {
  /*
   The background view square is rendered to the corners
   of the world coordinates.
   */
  var squarePosition = [
    [props.w()[0], props.w()[1], 1],
    [props.w()[4], props.w()[5], 1],
    [props.w()[8], props.w()[9], 1],
    [props.w()[12], props.w()[13], 1]
  ];

  var squareUv = [
    /*
     Texture coordinates for the background view.
     */
    [1.0, 0.0],
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],  // positive z face.
  ];
  const squareElements = [
    /*
     Drawing order of background view square.
     */
    [0, 1, 2],
    [0, 3, 2]  // positive z face.
  ];
  const drawSquare = regl({
    /*
     regl function to perform an openGL drawing operation.
     Draws the background view square onto a texture.
     */
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
  vec3 pos = position;
  gl_Position = projection * view * vec4(pos, 1);
}`,
    attributes: {position: squarePosition, uv: squareUv},
    elements: squareElements,
    uniforms: {
      projection: ({viewportWidth, viewportHeight}) => getBackgroundProjectionMatrix(props),
      view: ({tick}, props)                         => getBackgroundViewMatrix(props.props),
      tex: regl.prop('data')
    }
  })

  /*
   Load an image from the public folder, then render it into an invisible div.
   Capture the imageData and then pass that to regl to render as a texture.
   */
  /*
   Load a hidden image into the backgroundCanvas div
   */
  const backgroundCanvas = document.createElement('div');
  backgroundCanvas.innerHTML =
    '<canvas id=\'hiddenBackground\' width=3000 height=3000 style=\'visibility: hidden\'/>'
  const image = new Image();
  image.src   = './usa_map.png';
  image.style = 'visibility: hidden';
  backgroundCanvas.appendChild(image);
  document.body.appendChild(backgroundCanvas);
  const canvas = document.getElementById('hiddenBackground');
  /*
   When the image is finished loading
   */
  image.onload = () => {
    /*
     * Draw the image to a context
     */
    const context = canvas.getContext('2d');
    context.drawImage(image, 0, 0);
    const imWidth  = image.width;
    const imHeight = image.height;
    /*
     Get a copy of the image data in the canvas.
     */
    const imageData = context.getImageData(0, 0, imWidth, imHeight);
    /*
     Create a regl texture from the image data
     */
    const data = regl.texture({width: imWidth, height: imHeight, data: imageData.data});
    const tick = regl.frame(() => {
      /*
       Render the background.
       */
      regl.clear({depth: 1, color: [0, 0, 0, 0]});
      drawSquare({data, props});
    });
  }
}
