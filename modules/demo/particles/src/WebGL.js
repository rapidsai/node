/**
 * Copyright (c) Mik BRY
 * mik@miklabs.com
 *
 * This source code is licensed under private license found in the
 * LICENSE file in the root directory of this source tree.
 */

import { initShaderProgram } from './utils/shaderTools';
import { perspective } from './utils/matrix4';

export default class WebGL {
  constructor(canvas, width, height) {
    this.canvas = canvas;
    this.width = width;
    this.height = height;
  }

  init(scene) {
    const gl = this.canvas.getContext('webgl');
    if (!gl) {
      throw new Error('WebGL : unable to initialize');
    }
    this.gl = gl;
    this.scene = scene;

    this.buildShaders(scene.shaders);
    this.buildBuffers(scene.model);
    scene.init(this);
    this.initProjection(gl);
    this.render = this.render.bind(this);
    this.requestId = requestAnimationFrame(this.render);
  }

  initProjection(gl) {
    const fieldOfView = (45 * Math.PI) / 180; // in radians
    const aspect = gl.canvas.clientWidth / gl.canvas.clientHeight;
    const zNear = 0.1;
    const zFar = 100.0;
    this.projectionMatrix = perspective(fieldOfView, aspect, zNear, zFar);
  }

  render(now) {
    this.scene.render(this, now);
    requestAnimationFrame(this.render);
  }

  buildShaders(shaders) {
    const { gl } = this;
    const shaderProgram = initShaderProgram(gl, shaders.vs, shaders.fs);
    this.programInfo = {
      program: shaderProgram,
      attribLocations: {
        vertexPosition: gl.getAttribLocation(shaderProgram, 'aVertexPosition'),
        vertexColor: gl.getAttribLocation(shaderProgram, 'aVertexColor'),
      },
      uniformLocations: {
        projectionMatrix: gl.getUniformLocation(shaderProgram, 'uProjectionMatrix'),
        modelViewMatrix: gl.getUniformLocation(shaderProgram, 'uModelViewMatrix'),
      },
    };
  }

  buildBuffers({ positions, colors }) {
    const { gl } = this;
    const positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);
    const colorBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(colors), gl.STATIC_DRAW);

    this.buffers = {
      position: positionBuffer,
      color: colorBuffer,
    };
  }

  close() {
    cancelAnimationFrame(this.requestId);
  }
}
