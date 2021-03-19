// Copyright (c) 2020, NVIDIA CORPORATION.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import {Buffer} from '../../buffer';

export const nodePositionAccessor = (gl: WebGLRenderingContext) => ({size: 1, type: gl.FLOAT});

export const nodeColorAccessor = (gl: WebGLRenderingContext) =>
  ({size: 4, type: gl.UNSIGNED_BYTE, normalized: true});

export const nodeRadiusAccessor = (gl: WebGLRenderingContext) =>
  ({size: 1, type: gl.UNSIGNED_BYTE});

export const nodeElementIndicesAccessor = (gl: WebGLRenderingContext) =>
  ({size: 1, type: gl.UNSIGNED_INT});

export class NodeColorBuffer extends Buffer {
  constructor(gl: WebGLRenderingContext, byteLength = 0) {
    byteLength = Math.max(byteLength || 0, 1);
    super(gl, {byteLength, accessor: nodeColorAccessor(gl)});
  }
}

export class NodeRadiusBuffer extends Buffer {
  constructor(gl: WebGLRenderingContext, byteLength = 0) {
    byteLength = Math.max(byteLength || 0, 1);
    super(gl, {byteLength, accessor: nodeRadiusAccessor(gl)});
  }
}

export class NodePositionBuffer extends Buffer {
  constructor(gl: WebGLRenderingContext, byteLength = 0) {
    byteLength = Math.max(byteLength || 0, 1);
    super(gl, {byteLength, accessor: nodePositionAccessor(gl)});
  }
}

export class NodeElementIndicesBuffer extends Buffer {
  constructor(gl: WebGLRenderingContext, byteLength = 0) {
    byteLength = Math.max(byteLength || 0, 1);
    super(gl, {byteLength, accessor: nodeElementIndicesAccessor(gl)});
  }
}
