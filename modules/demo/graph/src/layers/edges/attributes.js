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

import { Buffer } from '@rapidsai/deck.gl';
import { Texture2D } from '@luma.gl/webgl';

export const edgeListAccessor = (gl) => ({ size: 2, type: gl.UNSIGNED_INT });
export const edgeComponentAccessor = (gl) => ({ size: 3, type: gl.FLOAT });
export const edgeSourceColorAccessor = (gl) => ({ size: 4, stride: 8, offset: 0, type: gl.UNSIGNED_BYTE });
export const edgeTargetColorAccessor = (gl) => ({ size: 4, stride: 8, offset: 4, type: gl.UNSIGNED_BYTE });

export class EdgeListBuffer extends Buffer {
  constructor(gl, byteLength = 0) {
    byteLength = Math.max(byteLength || 0, 1);
    super(gl, { byteLength, accessor: edgeListAccessor(gl) });
  }
}

export class EdgeColorBuffer extends Buffer {
  constructor(gl, byteLength = 0) {
    byteLength = Math.max(byteLength || 0, 1);
    super(gl, { byteLength, accessor: { ...edgeSourceColorAccessor(gl), size: 8 } });
  }
}

export class EdgeComponentBuffer extends Buffer {
  constructor(gl, byteLength = 0) {
    byteLength = Math.max(byteLength || 0, 1);
    super(gl, { byteLength, accessor: edgeComponentAccessor(gl) });
  }
}

// Transform feedback buffers
export class EdgePositionTexture extends Texture2D {
  constructor(gl) {
    super(gl, {
      format: gl.R32F,
      type: gl.FLOAT,
      width: 1,
      height: 1,
      parameters: {
        [gl.TEXTURE_MIN_FILTER]: [gl.NEAREST],
        [gl.TEXTURE_MAG_FILTER]: [gl.NEAREST],
        [gl.TEXTURE_WRAP_S]: [gl.CLAMP_TO_EDGE],
        [gl.TEXTURE_WRAP_T]: [gl.CLAMP_TO_EDGE],
      },
      mipmap: false
    })
  }
}
