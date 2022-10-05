/**
 * Copyright (c) Mik BRY
 * mik@mikbry.com
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

import { create, translate, rotate } from './utils/matrix4';
import fs from './shaders/one.fs';
import vs from './shaders/one.vs';

const scene = {
  shaders: {
    fs,
    vs,
  },
  model: {
    positions: [1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0],
    colors: [
      1.0,
      1.0,
      1.0,
      1.0, // white
      1.0,
      0.0,
      0.0,
      1.0, // red
      0.0,
      1.0,
      0.0,
      1.0, // green
      0.0,
      0.0,
      1.0,
      1.0, // blue
    ],
  },
};

const impl = {};

scene.init = engine => {
  impl.engine = engine;
  impl.squareRotation = 0;
  impl.from = null;
};

scene.render = (engine, now) => {
  const { gl, programInfo, buffers } = engine;

  gl.clearColor(0.0, 0.0, 0.0, 0.0);
  gl.clearDepth(1.0);
  gl.enable(gl.DEPTH_TEST);
  gl.depthFunc(gl.LEQUAL);
  // eslint-disable-next-line no-bitwise
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  const modelViewMatrix = create();
  translate(modelViewMatrix, modelViewMatrix, [-0.0, 0.0, -6.0]);
  rotate(modelViewMatrix, modelViewMatrix, -impl.squareRotation, [0, 0, 1]);

  gl.bindBuffer(gl.ARRAY_BUFFER, buffers.position);
  gl.vertexAttribPointer(programInfo.attribLocations.vertexPosition, 2, gl.FLOAT, false, 0, 0);
  gl.enableVertexAttribArray(programInfo.attribLocations.vertexPosition);
  gl.bindBuffer(gl.ARRAY_BUFFER, buffers.color);
  gl.vertexAttribPointer(programInfo.attribLocations.vertexColor, 4, gl.FLOAT, false, 0, 0);
  gl.enableVertexAttribArray(programInfo.attribLocations.vertexColor);
  gl.useProgram(programInfo.program);
  gl.uniformMatrix4fv(programInfo.uniformLocations.projectionMatrix, false, engine.projectionMatrix);
  gl.uniformMatrix4fv(programInfo.uniformLocations.modelViewMatrix, false, modelViewMatrix);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

  const deltaTime = impl.from === null ? 0 : now - impl.from;
  impl.squareRotation += deltaTime * 0.001;
  impl.from = now;
};

export default scene;
